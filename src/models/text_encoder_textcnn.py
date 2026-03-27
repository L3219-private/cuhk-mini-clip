# src/models/text_encoder_textcnn.py

import torch
import torch.nn as nn


class TextEncoder_TextCNN(nn.Module):
    """
    TextCNN encoder that maps padded token IDs to CLIP embedding space.
    - embed_dim: Output dimension — must match the image encoder's embed_dim.
    - word_dim: dim of the word embedding lookup table
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        word_dim: int = 128,
        n_filters: int = 100,
        kernel_sizes: tuple = (2, 3, 4, 5),
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_dim,
            padding_idx=pad_idx,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=n_filters,
                kernel_size=k,
                padding=0,  # since max-pool immediately
            )
            for k in kernel_sizes
        ])

        # projection
        total_filters = n_filters * len(kernel_sizes)  # e.g. 100*4=400
        self.projection = nn.Sequential(
            nn.Linear(total_filters, total_filters),
            nn.ReLU(inplace=True),
            nn.Linear(total_filters, embed_dim),
        )

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """text_ids: (B, L) tensor of padded token IDs."""
        x = self.embedding(text_ids)  # (B, L) → (B, L, word_dim)

        x = x.transpose(1, 2)  # (B, word_dim, L) to be input in conv1d

        # Apply each conv + ReLU + max-pool-over-time
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))        # (B, n_filters, L - k + 1)
            h = h.max(dim=2).values        # (B, n_filters)
            pooled.append(h)

        # Concatenate all kernel results
        cat = torch.cat(pooled, dim=1)     # (B, n_filters * num_kernels)

        return self.projection(cat)        # (B, embed_dim)
