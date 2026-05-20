import torch
from model.task_heads import TaskHead

head = TaskHead()
x = torch.zeros(2, 800)
output = head(x)
print('Output shape:', output.shape)
print('Output values:', output)