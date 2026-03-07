# src/datasets/custom.py

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
from PIL import Image
import torch

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}

def resolve_img_path(images_dir: Path, img_rel: str) -> Path:
    """turn all the paths to be absolute or start with data/"""
    p = Path(img_rel.strip())
    posix = p.as_posix()  # '\' -> '/'
    if posix.startswith(("data/")) or p.is_absolute():
        return p
    return images_dir / p


def is_image_file(path: Path) -> bool:
    """check if the file has an allowed image extension"""
    return path.suffix.lower() in ALLOWED_EXTS


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of dicts with 'image' and 'caption' keys."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "image" in obj and "caption" in obj:
                    rows.append(obj)
            except Exception:
                continue
    return rows


class Vocabtable:
    PAD = "<pad>"  # padding, enable batch training
    UNK = "<unk>"  # unknown token
    
    # build dictionary
    def __init__(self, tokens: list[str]):  # first entry
        toks = [self.PAD, self.UNK] + sorted(set(tokens))
        self.stoi = {s: i for i, s in enumerate(toks)}
        self.itos = {i: s for s, i in self.stoi.items()}

    @classmethod
    def build_from_captions(cls, captions: list[str], max_tokens: int = 20000):  # second entry
        """extract the most common max_tokens tokens split by whitespace to build the vocab table"""
        from collections import Counter
        cnt = Counter()  # a Counter to count token frequencies
        for cap in captions:
            tok = cap.strip().split()
            cnt.update(tok)
        most = [s for s, _ in cnt.most_common(max_tokens)]
        return cls(most)  # extract the most common max_tokens tokens to build the vocab table

    # search in dictionary
    def encode(self, text: str) -> list[int]:
        """encode the text into a list of token ids, using UNK id for OOV tokens"""
        return [self.stoi.get(s, self.stoi[self.UNK]) for s in text.strip().split()]

    def pad_id(self) -> int:
        return self.stoi[self.PAD]


@dataclass
class Sample:
    image: torch.Tensor  # (C,H,W) float32 [0,1]
    text_ids: torch.Tensor  # (L,) long
    text_len: int
    raw_caption: str


class CustomDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        jsonl_path: str,
        vocab: Optional[Vocabtable] = None,
        image_size: Tuple[int, int] = (224, 224),  # (H, W)
    ):
        self.images_dir = Path(images_dir)
        self.jsonl_path = Path(jsonl_path)
        self.rows = load_jsonl(self.jsonl_path)  # dict
        self.image_size = image_size

        if vocab is None:
            captions = [r["caption"] for r in self.rows]
            self.vocab = Vocabtable.build_from_captions(captions, max_tokens=20000)
        else:
            self.vocab = vocab


    def __len__(self) -> int:
        return len(self.rows)

    def load_image_tensor(self, img_path: Path) -> torch.Tensor:
        """load one image and convert to (3, H, W) float 32 tensor in [0, 1]"""
        if not is_image_file(img_path):
            raise ValueError(f"Not an image file (ext filtered): {img_path}")
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            # PIL resize takes (W,H); we store (H,W) in image_size (need to reverse)
            im = im.resize((self.image_size[1], self.image_size[0]))
            # 1D tensor of bytes
            buf = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
            # reshape to (H,W,3)
            buf = buf.view(self.image_size[0], self.image_size[1], 3)
            # normalize to [0,1] and permute to (3, H, W)
            x = buf.permute(2, 0, 1).contiguous().float() / 255.0
            return x

    def __getitem__(self, idx: int) -> Sample:
        row = self.rows[idx]
        img_rel = row["image"]
        cap = row["caption"]
        img_path = resolve_img_path(self.images_dir, img_rel)
        image = self.load_image_tensor(img_path)
        text_ids = torch.tensor(self.vocab.encode(cap), dtype=torch.long)
        return Sample(image=image, text_ids=text_ids, text_len=int(text_ids.numel()), raw_caption=cap)


def pad_sequence(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
    """Pad a list of 1D tensors to the same length, return a 2D tensor"""
    if len(seqs) == 0:
        return torch.empty(0, 0, dtype=torch.long)
    L = max(int(s.numel()) for s in seqs)  # max text length (how many tokens)
    out = torch.full((len(seqs), L), pad_val, dtype=torch.long)
    for i, s in enumerate(seqs):
        l = int(s.numel())
        out[i, :l] = s
    return out


def collate(batch: List[Sample], pad_id: int) -> Dict[str, torch.Tensor]:
    """combine Samples into a batch dictionary"""
    images = torch.stack([b.image for b in batch], dim=0)  # (B,3,H,W)
    lens = torch.tensor([b.text_len for b in batch], dtype=torch.long)
    text = pad_sequence([b.text_ids for b in batch], pad_val=pad_id)  # (B,L)
    return {"images": images, "text": text, "text_len": lens}


