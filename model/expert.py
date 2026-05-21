import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_DIM, FFN_UNITS, DROPOUT, NUM_HEADS

class ExpertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # manual attention projections
        self.W_q = nn.Linear(INPUT_DIM, INPUT_DIM)
        self.W_k = nn.Linear(INPUT_DIM, INPUT_DIM)
        self.W_v = nn.Linear(INPUT_DIM, INPUT_DIM)
        self.W_o = nn.Linear(INPUT_DIM, INPUT_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        
        self.ffn = nn.Sequential(
            nn.Linear(INPUT_DIM, FFN_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_UNITS, FFN_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        self.scale = (INPUT_DIM // NUM_HEADS) ** 0.5

    def forward(self, x):
        # x: [batch, seq, dim]
        Q = self.W_q(x)  # [batch, seq, dim]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # scaled dot product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch, seq, seq]
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        x = torch.bmm(scores, V)  # [batch, seq, dim]
        x = self.W_o(x)
        
        # FFN
        x = self.ffn(x)
        
        # dual pooling
        max_pool = torch.max(x, dim=1).values
        avg_pool = torch.mean(x, dim=1)
        return torch.cat([max_pool, avg_pool], dim=-1)