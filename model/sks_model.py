import torch.nn as nn
from model.input_layer import InputLayer
from model.mmoe import Mmoe
from model.task_heads import TaskHead

class SksModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer=InputLayer()
        self.mmoe=Mmoe()
        self.hate_head= TaskHead()
        self.sentiment_head= TaskHead()

    def forward(self, tweets, word_category_ids):
        x=self.input_layer(tweets, word_category_ids)
        hate_repr, sentiment_repr=self.mmoe(x)
        hate_res=self.hate_head(hate_repr)
        sentiment_res=self.sentiment_head(sentiment_repr)
        return hate_res,sentiment_res


