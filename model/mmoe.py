import torch
import torch.nn as nn
from config import NUM_EXPERTS,INPUT_DIM
from model.expert import ExpertLayer

class Mmoe(nn.Module):
    def __init__ (self):
        super().__init__()
        self.experts = nn.ModuleList([ExpertLayer() for _ in range (NUM_EXPERTS)])
        self.gates = nn.ModuleList([nn.Linear(INPUT_DIM, NUM_EXPERTS) for _ in range (2)])

    def forward(self,x):
        expert_outputs = [expert(x) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)
        pooled = x.mean(dim=1)  # [batch, 400]

        # gate for task 1 (hate speech)
        gate0_scores = torch.softmax(self.gates[0](pooled), dim=-1).unsqueeze(-1)  # [batch, 4, 1]
        task0_output = (gate0_scores * expert_stack).sum(dim=1) # [batch, 800]

        # gate for task 2 (sentiment)
        gate1_scores = torch.softmax(self.gates[1](pooled), dim=-1).unsqueeze(-1) # [batch, 4, 1]
        task1_output = (gate1_scores * expert_stack).sum(dim=1) # [batch, 800]

        return task0_output, task1_output
        
        

