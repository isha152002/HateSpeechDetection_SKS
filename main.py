import torch
from model.sks_model import SksModel

model = SksModel()

# fake batch of 2 tweets
glove = torch.zeros(2, 128, 300)
categories = torch.zeros(2, 128).long()

hate_pred, sentiment_pred = model(glove, categories)
print('Hate pred shape:', hate_pred.shape)
print('Sentiment pred shape:', sentiment_pred.shape)
print('Hate pred values:', hate_pred)
print('Sentiment pred values:', sentiment_pred)