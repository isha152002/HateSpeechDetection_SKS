from config import CATEGORY_EMB_DIM
import torch
import torch.nn as nn

class InputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.category_embedding = nn.Embedding(num_embeddings=2,embedding_dim=CATEGORY_EMB_DIM)

    def forward(self, glove_tensor, category_tensor):
        cat_emb = self.category_embedding(category_tensor)
        return torch.cat([glove_tensor, cat_emb], dim=-1)

