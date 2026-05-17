import torch
from model.expert import ExpertLayer

expert = ExpertLayer()

# fake batch of 2 tweets, 128 tokens, 400 dims
x = torch.zeros(2, 128, 400)

output = expert(x)
print('Output shape:', output.shape)