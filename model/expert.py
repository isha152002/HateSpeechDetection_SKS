import torch
import torch.nn as nn
from config import INPUT_DIM, FFN_UNITS, DROPOUT,NUM_HEADS

class ExpertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=INPUT_DIM, num_heads=NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(INPUT_DIM, FFN_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_UNITS, FFN_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

    def forward(self,x):
        # use torch.backends to fix compatibility issue
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            x, _ = self.attention(x, x, x)
        x = self.ffn(x)
        max_pool = torch.max(x, dim=1).values
        avg_pool = torch.mean(x, dim=1)
        return torch.cat([max_pool, avg_pool], dim=-1)
