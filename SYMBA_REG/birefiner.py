import torch
import torch.nn as nn


class BiRefiner(nn.Module):
    def __init__(self, vocab: int, emb_dim: int = 96, gpt_dim: int = 192,
                 layers: int = 2, heads: int = 4, max_len: int = 60):
        super().__init__()
        self.tok_emb  = nn.Embedding(vocab, gpt_dim)
        self.pos_emb  = nn.Embedding(max_len + 2, gpt_dim)
        self.ctx_proj = nn.Linear(emb_dim, gpt_dim)
        self.mask_emb = nn.Parameter(torch.randn(gpt_dim) * 0.02)
        enc_layer     = nn.TransformerEncoderLayer(
            gpt_dim, heads, 4 * gpt_dim, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Linear(gpt_dim, vocab)

    def forward(self, tids: torch.Tensor, emb: torch.Tensor,
                mask_positions=None) -> torch.Tensor:
        B, T = tids.shape
        pos  = torch.arange(T, device=tids.device)
        x    = self.tok_emb(tids) + self.pos_emb(pos).unsqueeze(0)
        if mask_positions is not None:
            for b in range(B):
                for p in mask_positions[b]:
                    if p < T:
                        x[b, p] = self.mask_emb
        ctx = self.ctx_proj(emb).unsqueeze(1)
        x   = torch.cat([ctx, x], dim=1)
        x   = self.transformer(x)
        return self.out(x[:, 1:, :])
