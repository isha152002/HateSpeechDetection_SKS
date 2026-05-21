import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_DIM, FFN_UNITS, DROPOUT, NUM_HEADS

class ExpertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = INPUT_DIM // NUM_HEADS  # 400 // 4 = 100
        
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

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # project Q, K, V
        Q = self.W_q(x)  # [batch, seq, 400]
        K = self.W_k(x)
        V = self.W_v(x)

        # split into NUM_HEADS heads
        Q = Q.view(batch_size, seq_len, NUM_HEADS, self.head_dim).transpose(1, 2)  # [batch, heads, seq, head_dim]
        K = K.view(batch_size, seq_len, NUM_HEADS, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, NUM_HEADS, self.head_dim).transpose(1, 2)

        # scaled dot product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, heads, seq, seq]
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        # weighted sum of values
        x = torch.matmul(scores, V)  # [batch, heads, seq, head_dim]

        # merge heads back
        x = x.transpose(1, 2).contiguous()  # [batch, seq, heads, head_dim]
        x = x.view(batch_size, seq_len, INPUT_DIM)  # [batch, seq, 400]

        # output projection
        x = self.W_o(x)

        # FFN
        x = self.ffn(x)

        # dual pooling
        max_pool = torch.max(x, dim=1).values
        avg_pool = torch.mean(x, dim=1)
        return torch.cat([max_pool, avg_pool], dim=-1)