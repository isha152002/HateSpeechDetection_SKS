import torch 
import torch.nn as nn
from config import EXPERT_OUT_DIM

class TaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(EXPERT_OUT_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x=self.linear(x)
        x=self.sigmoid(x)
        return x