"""
data_loader.py
--------------
Handles loading, preprocessing and batching of all three datasets:
  - HateSpeechDataset: SemEval-2019 (parquet) and Davidson (csv)
  - SentimentDataset:  Kaggle Twitter Sentiment (csv)

Each dataset returns per sample:
  - glove_tensor:    [128 x 300] float32  — GloVe word vectors, padded/truncated
  - category_tensor: [128]       long     — 0=non-toxic, 1=toxic per token
  - label_tensor:    scalar      long     — ground truth label (0 or 1)

Usage:
  glove_emb = load_glove('data/raw/glove.840B.300d.txt')
  derog_dict = load_dictionary('data/raw/derogatory_dictionary.csv')
  dataset = HateSpeechDataset('data/raw/davidson.csv', glove_emb, derog_dict, 'davidson')
  loader  = create_dataloader(dataset, shuffle=True)
"""

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from utils.preprocessing import clean_tweet
from utils.dictionary import get_category
from config import MAX_SEQ_LEN,BATCH_SIZE
import numpy as np


def load_glove(filepath):
    """
    Load GloVe embeddings from text file into a dictionary.
    Args:
        filepath: path to glove.840B.300d.txt (2.2GB file)
    Returns:
        dict: {word: np.array of shape (300,)}

    """
    glove_embeddings={}
    with open(filepath, 'r',encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:],dtype=np.float32)
            glove_embeddings[word] = vector
    return glove_embeddings


class HateSpeechDataset(Dataset):
    def __init__(self,filepath,glove_embeddings,derogatory_dict,dataset_type):
        self.dataset_type=dataset_type
        if dataset_type=="semeval":
            df=pd.read_parquet(filepath)
            df = df[df['language'] == 'en']  # filter English only
            texts = df['text'].tolist() # get tweet column as list
            labels = df['HS'].tolist()  # get label column as list
        elif dataset_type == "davidson":
            df=pd.read_csv(filepath)
            texts = df['tweet'].tolist()
            labels = [1 if x == 0 else 0 for x in df['class'].tolist()]
            # class 0 = hate → label 1
            # class 1,2 = not hate → label 0

        self.glove_embeddings=glove_embeddings
        self.derogatory_dict=derogatory_dict
        self.tweets=[]
        self.labels=[]

        for tweet, label in zip(texts, labels):
            self.tweets.append(clean_tweet(tweet))
            self.labels.append(label)


    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self,idx): # Convert a cleaned tweet string into model-ready tensors.
        tweet= self.tweets[idx]
        label=self.labels[idx]

        #step1: tokenise
        tokens=tweet.split()

        #step2: glove lookup per token
        glove_vectors=[]
        category_ids=[]

        for token in tokens:
            if token in self.glove_embeddings:
                glove_vectors.append(self.glove_embeddings[token])
            else:
                glove_vectors.append(np.zeros(300,dtype=np.float32))

            #category lookup
            category=get_category(token,self.derogatory_dict)
            category_ids.append(0 if category=="C0" else 1)

        #step3: pad or truncate to 128
        # truncate if too long
        glove_vectors = glove_vectors[:MAX_SEQ_LEN]
        category_ids = category_ids[:MAX_SEQ_LEN]

        # pad if too short
        while len(glove_vectors) < MAX_SEQ_LEN:
            glove_vectors.append(np.zeros(300, dtype=np.float32))
            category_ids.append(0)  # pad category with C0

        # Step 4: convert to tensors
        glove_tensor = torch.tensor(np.array(glove_vectors), dtype=torch.float32)
        category_tensor = torch.tensor(category_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return glove_tensor, category_tensor, label_tensor


class SentimentDataset(Dataset):
    def __init__(self,filepath,glove_embeddings,derogatory_dict):
        df=pd.read_csv(filepath)
        texts=df['tweet'].tolist()
        labels=df['label'].tolist()
        self.glove_embeddings=glove_embeddings
        self.derogatory_dict=derogatory_dict
        self.tweets=[]
        self.labels=[]

        for tweet, label in zip(texts, labels):
            self.tweets.append(clean_tweet(tweet))
            self.labels.append(label)

    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self,idx):
        tweet= self.tweets[idx]
        label=self.labels[idx]

        #step1: tokenise
        tokens=tweet.split()

        #step2: glove lookup per token
        glove_vectors=[]
        category_ids=[]

        for token in tokens:
            if token in self.glove_embeddings:
                glove_vectors.append(self.glove_embeddings[token])
            else:
                glove_vectors.append(np.zeros(300,dtype=np.float32))

            #category lookup
            category=get_category(token,self.derogatory_dict)
            category_ids.append(0 if category=="C0" else 1)

        #step3: pad or truncate to 128
        # truncate if too long
        glove_vectors = glove_vectors[:MAX_SEQ_LEN]
        category_ids = category_ids[:MAX_SEQ_LEN]

        # pad if too short
        while len(glove_vectors) < MAX_SEQ_LEN:
            glove_vectors.append(np.zeros(300, dtype=np.float32))
            category_ids.append(0)  # pad category with C0

        # Step 4: convert to tensors
        glove_tensor = torch.tensor(np.array(glove_vectors), dtype=torch.float32)
        category_tensor = torch.tensor(category_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return glove_tensor, category_tensor, label_tensor

def create_dataloader(dataset, shuffle=True):
    """
    Wrap a Dataset in a PyTorch DataLoader for batching.
    Args:
        dataset: HateSpeechDataset or SentimentDataset
        shuffle: True during training, False during evaluation
    Returns:
        DataLoader yielding batches of (glove_tensor, category_tensor, label_tensor)
    """
    return DataLoader(dataset,batch_size=BATCH_SIZE, shuffle=shuffle)        