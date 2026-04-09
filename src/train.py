# training script, run with: python -m src.train

import argparse
import json
import os
import random
import time
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.datasets.custom import CustomDataset, Vocabtable, collate
from src.models.image_encoder_smallcnn import ImageEncoder_SmallCNN
from src.models.image_encoder_resnet18 import ImageEncoder_ResNet18
from src.models.image_encoder_resnet50 import ImageEncoder_ResNet50
from src.models.text_encoder_textcnn import TextEncoder_TextCNN
from src.models.text_encoder_bigru import TextEncoder_BiGRU
from src.utils.device import pick_device
from src.utils.param_stats import print_param_stats


def set_seed(seed: int):
    # fix random seeds so results are reproducible across runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_image_encoder(name: str, embed_dim: int) -> nn.Module:
    # all image encoders: (B,3,H,W) -> (B, embed_dim)
    name = name.lower()
    if name == "smallcnn":
        return ImageEncoder_SmallCNN(embed_dim=embed_dim)
    elif name == "resnet18":
        return ImageEncoder_ResNet18(embed_dim=embed_dim)
    elif name == "resnet50":
        return ImageEncoder_ResNet50(embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown image encoder: {name}")


def build_text_encoder(name: str, vocab_size: int, embed_dim: int,
                       model_cfg: dict) -> nn.Module:
    # all text encoders: (B,L) -> (B, embed_dim)
    name = name.lower()
    if name == "textcnn":
        tcfg = model_cfg.get("textcnn", {})
        return TextEncoder_TextCNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            word_dim=tcfg.get("word_dim", 128),
            n_filters=tcfg.get("n_filters", 100),
            kernel_sizes=tuple(tcfg.get("kernel_sizes", [2, 3, 4, 5])),
        )
    elif name == "bigru":
        bcfg = model_cfg.get("bigru", {})
        return TextEncoder_BiGRU(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            word_dim=bcfg.get("word_dim", 128),
            hidden_dim=bcfg.get("hidden_dim", 128),
            num_layers=bcfg.get("num_layers", 2),
            dropout=bcfg.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown text encoder: {name}")


# contrastive loss (symmetric InfoNCE)
# logits[i][j] = sim(image_i, text_j) / temp
# correct pairs are on the diagonal, cross-entropy pushes them to score highest
def clip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor,
              temperature: float) -> torch.Tensor:
    # temperature can be a plain float or a learnable nn.Parameter (log scale)
    t = temperature.exp() if isinstance(temperature, torch.Tensor) else temperature
    logits = (img_emb @ txt_emb.T) / t                # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)        # image → find text
    loss_t2i = F.cross_entropy(logits.T, labels)      # text → find image

    return (loss_i2t + loss_t2i) / 2

def train_one_epoch(image_encoder, text_encoder, loader,
                    optimizer, device, temperature, max_grad_norm=1.0):
    # max_grad_norm: gradient clipping, helps keep BiGRU stable
    image_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    n_batches = 0

    num_batches = len(loader)
    log_every = max(num_batches // 20, 1)       # print ~20 times per epoch

    for i, batch in enumerate(loader):
        if i == 0:
            print(f"    [batch 1/{num_batches}] loading first batch (JIT warmup may be slow)...", flush=True)

        images = batch["images"].to(device)  # (B, 3, H, W)
        text   = batch["text"].to(device)  # (B, L)

        # forward: encode + L2 normalise
        img_emb = F.normalize(image_encoder(images), dim=-1)
        txt_emb = F.normalize(text_encoder(text),    dim=-1)

        loss = clip_loss(img_emb, txt_emb, temperature)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping for training stability (especially BiGRU)
        all_params = list(image_encoder.parameters()) + list(text_encoder.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (i + 1) % log_every == 0 or (i + 1) == num_batches:
            avg = total_loss / n_batches
            print(f"    [batch {i+1}/{num_batches}]  loss={avg:.4f}", flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(image_encoder, text_encoder, loader, device, temperature):
    # compute val loss without updating weights
    image_encoder.eval()
    text_encoder.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device)
        text   = batch["text"].to(device)

        img_emb = F.normalize(image_encoder(images), dim=-1)
        txt_emb = F.normalize(text_encoder(text),    dim=-1)

        loss = clip_loss(img_emb, txt_emb, temperature)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # set seed
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    # choose device
    device_cfg = cfg.get("device", {}).get("prefer", "auto")
    device = pick_device(device_cfg=device_cfg)
    print(f"[Info] Device: {device}")

    # load hyperparameters
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    embed_dim        = model_cfg.get("embed_dim", 128)
    temp_init        = train_cfg.get("temperature", 0.07)
    temp_mode        = train_cfg.get("temperature_mode", "fixed")   # fixed | learn
    batch_size       = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 20)
    lr = train_cfg.get("lr", 1e-4)
    patience = train_cfg.get("patience", 5)

    # load dataset
    paths = cfg.get("paths", {})
    images_dir  = paths.get("images", "data/flickr30k/images")
    train_jsonl = paths.get("train_list", "data/flickr30k/train.jsonl")
    val_jsonl   = paths.get("val_list", "data/flickr30k/val.jsonl")

    print(f"[Info] Loading training data ...")
    train_dataset = CustomDataset(images_dir=images_dir, jsonl_path=train_jsonl)
    vocab = train_dataset.vocab
    pad_id = vocab.pad_id()

    print(f"[Info] Loading validation data ...")
    val_dataset = CustomDataset(
        images_dir=images_dir, jsonl_path=val_jsonl, vocab=vocab
    )

    collate_fn = partial(collate, pad_id=pad_id)

    # drop_last=True: CLIP needs a full batch, a tiny last batch has too few negatives
    n_workers = min(8, len(os.sched_getaffinity(0)))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=n_workers, drop_last=True,
        persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers,
        persistent_workers=(n_workers > 0),
    )

    vocab_size = len(vocab.stoi)
    print(f"[Info] Vocab size: {vocab_size}")
    print(f"[Info] Train: {len(train_dataset)} pairs | Val: {len(val_dataset)} pairs")

    # build models
    img_enc_name = model_cfg.get("image_encoder", "smallcnn")
    txt_enc_name = model_cfg.get("text_encoder", "textcnn")

    image_encoder = build_image_encoder(img_enc_name, embed_dim).to(device)
    text_encoder  = build_text_encoder(
        txt_enc_name, vocab_size, embed_dim, model_cfg
    ).to(device)

    print_param_stats(image_encoder, f"Image Encoder ({img_enc_name})")
    print_param_stats(text_encoder,  f"Text Encoder ({txt_enc_name})")

    # temperature: fixed = plain float, learn = nn.Parameter updated by optimizer
    # store log(temp) so temp stays positive
    if temp_mode == "learn":
        import math
        log_temp = nn.Parameter(torch.tensor(math.log(temp_init), device=device))
        temperature = log_temp   # pass the Parameter; clip_loss will call .exp()
        temp_params = [log_temp]
        print(f"[Info] Temperature: learnable (init={temp_init})")
    else:
        temperature = temp_init
        temp_params = []
        print(f"[Info] Temperature: fixed={temperature}")

    # optimizer: Adam over all parameters (both encoders + maybe log_temp)
    all_params = (list(image_encoder.parameters())
                  + list(text_encoder.parameters())
                  + temp_params)
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # training loop
    t_label = f"learn{temp_init}" if temp_mode == "learn" else f"fix{temp_init}"
    run_name = f"{img_enc_name}_{txt_enc_name}_d{embed_dim}_t{t_label}"
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 55}")
    print(f"  {run_name}")
    print(f"  batch_size={batch_size}  epochs={epochs}  lr={lr}")
    print(f"  patience={patience}  (early stopping)")
    print(f"{'=' * 55}\n")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            image_encoder, text_encoder, train_loader,
            optimizer, device, temperature,
        )

        val_loss = validate(
            image_encoder, text_encoder, val_loader,
            device, temperature,
        )

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>3}/{epochs} │ "
            f"train_loss {train_loss:.4f} │ "
            f"val_loss {val_loss:.4f} │ "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss":   round(val_loss, 5),
            "time_s":     round(elapsed, 1),
        })

        # save if val loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "image_encoder": image_encoder.state_dict(),
                "text_encoder":  text_encoder.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "val_loss":      val_loss,
                "config":        cfg,
            }, ckpt_dir / "best.pt")
            print(f"         └─ saved best (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stop] No improvement for {patience} epochs.")
                break

    # save history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 55}")
    print(f"  Finished: best val_loss = {best_val_loss:.4f}")
    print(f"  Checkpoint: {ckpt_dir}/best.pt")
    print(f"  History:    {ckpt_dir}/history.json")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()