import torch
import torch.nn as nn


class ARDecoder(nn.Module):
    def __init__(self, vocab: int, emb_dim: int = 96, gpt_dim: int = 192,
                 layers: int = 2, heads: int = 4,
                 max_len: int = 60, dropout: float = 0.20):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, gpt_dim)
        self.pos_emb = nn.Embedding(max_len + 2, gpt_dim)
        self.proj    = nn.Linear(emb_dim, gpt_dim)
        self.drop    = nn.Dropout(dropout)
        self.layers  = nn.ModuleList([nn.ModuleDict({
            'sa': nn.MultiheadAttention(gpt_dim, heads, batch_first=True, dropout=dropout),
            'ff': nn.Sequential(
                nn.Linear(gpt_dim, 4 * gpt_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(4 * gpt_dim, gpt_dim),
            ),
            'n1': nn.LayerNorm(gpt_dim),
            'n2': nn.LayerNorm(gpt_dim),
        }) for _ in range(layers)])
        self.out    = nn.Linear(gpt_dim, vocab)
        self._cache = {}

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        if T not in self._cache:
            self._cache[T] = torch.triu(
                torch.full((T, T), float('-inf'), device=device), diagonal=1
            )
        return self._cache[T]

    def forward(self, emb: torch.Tensor, tids: torch.Tensor) -> torch.Tensor:
        B, T = tids.shape
        pos  = torch.arange(T + 1, device=tids.device)
        x    = torch.cat([self.proj(emb).unsqueeze(1),
                          self.drop(self.tok_emb(tids))], dim=1)
        x    = x + self.pos_emb(pos).unsqueeze(0)
        mask = self._causal_mask(T + 1, tids.device)
        for l in self.layers:
            x = l['n1'](x + l['sa'](x, x, x, attn_mask=mask)[0])
            x = l['n2'](x + l['ff'](x))
        return self.out(x[:, 1:, :])
