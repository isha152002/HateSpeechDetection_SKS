import torch
from model.mmoe import Mmoe

mmoe = Mmoe()
x = torch.zeros(2, 128, 400)
task0, task1 = mmoe(x)
print('Task 0 shape:', task0.shape)
print('Task 1 shape:', task1.shape)