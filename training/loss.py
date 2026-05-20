import torch.nn as nn 
from config import ALPHA, BETA

def combined_loss(hate_pred, hate_label, sentiment_pred, sentiment_label):
    criterion=nn.BCELoss()
    hate_loss=criterion(hate_pred.squeeze(1),hate_label.float())
    sentiment_loss=criterion(sentiment_pred.squeeze(1),sentiment_label.float())
    return ALPHA*hate_loss + BETA*sentiment_loss