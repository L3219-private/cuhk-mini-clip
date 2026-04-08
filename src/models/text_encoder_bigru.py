# src/models/text_encoder_bigru.py

import torch
import torch.nn as nn


class TextEncoder_BiGRU(nn.Module):
    """
      token_ids (B, L)
        Embedding: (B, L, word_dim)
        2-layer BiGRU: (B, L, hidden_dim*2)
        masked mean pool: (B, hidden_dim*2)
        projection MLP: (B, embed_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        word_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_dim,
            padding_idx=pad_idx,
        )

        self.bigru = nn.GRU(
            input_size=word_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gru_out_dim, embed_dim),
        )

        self.pad_idx = pad_idx

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(text_ids)  # (B,L,word_dim)

        all_hidden, _ = self.bigru(x)  # (B,L,hidden*2)

        mask = (text_ids != self.pad_idx)  # (B,L)
        mask_f = mask.unsqueeze(-1).float()  # (B,L,1)
        masked = all_hidden * mask_f  # (B,L,hidden*2)
        lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (B,1)
        pooled = masked.sum(dim=1) / lengths  # (B,hidden*2)

        return self.projection(pooled)  # (B,embed_dim)
