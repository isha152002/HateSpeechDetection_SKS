from config import CATEGORY_EMB_DIM, MAX_SEQ_LEN
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class InputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokeniser = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base')
        self.roberta   = AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base')
        self.roberta.requires_grad_(False)  # frozen — don't update RoBERTa weights
        self.category_embedding = nn.Embedding(num_embeddings=2, embedding_dim=CATEGORY_EMB_DIM)  # learned vector

    def forward(self, tweets, word_category_ids):
        # instead of hardcoding cuda or cpu, ask the model itself which device it's on
        device = next(self.parameters()).device

        # STEP 1: tokenise the batch
        encoding = self.tokeniser(
            list(tweets),
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            is_split_into_words=False
        )
        input_ids      = encoding['input_ids'].to(device)       # [batch, 128]
        attention_mask = encoding['attention_mask'].to(device)  # [batch, 128]

        # STEP 2: align word-level category IDs to RoBERTa subword tokens
        if hasattr(word_category_ids, 'tolist'):
            word_category_ids = word_category_ids.tolist()
        batch_token_cats = []
        for i, word_cats in enumerate(word_category_ids):
            word_ids = encoding.word_ids(batch_index=i)
            # word_ids: [None, 0, 0, 1, 2, 2, None, None, ...]
            token_cats = []
            for word_id in word_ids:
                if word_id is None or word_id >= len(word_cats):
                    token_cats.append(0)  # special tokens and padding get C0
                else:
                    token_cats.append(word_cats[word_id])
            batch_token_cats.append(token_cats)

        category_tensor = torch.tensor(batch_token_cats, dtype=torch.long).to(device)  # [batch, 128]

        # STEP 3: run RoBERTa
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states  = roberta_output.last_hidden_state  # [batch, 128, 768]

        # STEP 4: category embedding lookup
        cat_emb = self.category_embedding(category_tensor)  # [batch, 128, 100]

        # STEP 5: concatenate and return
        return torch.cat([hidden_states, cat_emb], dim=-1)  # [batch, 128, 868]