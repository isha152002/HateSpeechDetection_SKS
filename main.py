import torch
from model.input_layer import InputLayer

layer = InputLayer()

# fake batch of 2 tweets
glove = torch.zeros(2, 128, 300)      # batch=2, seq=128, glove_dim=300
categories = torch.zeros(2, 128).long()  # batch=2, seq=128

output = layer(glove, categories)
print('Output shape:', output.shape)