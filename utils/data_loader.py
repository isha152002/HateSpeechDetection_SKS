"""
data_loader.py
--------------
Handles loading, preprocessing and batching of all three datasets:
  - HateSpeechDataset: SemEval-2019 (parquet) and Davidson (csv)
  - SentimentDataset:  Kaggle Twitter Sentiment (csv)

V3 change: GloVe lookup removed. Each sample now returns:
  - tweet:        str              — cleaned tweet string (tokenised by RoBERTa in input_layer)
  - category_ids: list of int      — word-level toxicity tags (0=non-toxic, 1=toxic)
                                     assigned before RoBERTa tokenisation so dictionary
                                     lookup works on whole words, not subword tokens
  - label:        int              — ground truth label (0 or 1)

Note: category_ids are word-level. Alignment to RoBERTa subword tokens
      happens in input_layer.py using tokeniser.word_ids()

Usage:
  derog_dict = load_dictionary('data/raw/derogatory_dictionary.csv')
  dataset = HateSpeechDataset('data/raw/davidson.csv', derog_dict, 'davidson')
  loader  = create_dataloader(dataset, shuffle=True)
"""

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from utils.preprocessing import clean_tweet_v3
from utils.dictionary import get_category
from config import MAX_SEQ_LEN,BATCH_SIZE
import numpy as np



class HateSpeechDataset(Dataset):
    def __init__(self,filepath,derogatory_dict,dataset_type):
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

        
        self.derogatory_dict=derogatory_dict
        self.tweets=[]
        self.category_ids=[]
        self.labels=[]

        for tweet, label in zip(texts, labels):
            cleaned=clean_tweet_v3(tweet)
            words=cleaned.split()
            word_cats = [1 if get_category(w, self.derogatory_dict) == "C1" else 0 for w in words]
            self.tweets.append(cleaned)
            self.category_ids.append(word_cats)
            self.labels.append(label)


    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self,idx): 
        return self.tweets[idx], self.category_ids[idx], self.labels[idx]


class SentimentDataset(Dataset):
    def __init__(self,filepath,derogatory_dict):
        df=pd.read_csv(filepath)
        texts=df['tweet'].tolist()
        labels=df['label'].tolist()
        self.derogatory_dict=derogatory_dict
        self.tweets=[]
        self.category_ids = []
        self.labels=[]

        for tweet, label in zip(texts, labels):
            cleaned = clean_tweet_v3(tweet)
            words = cleaned.split()
            word_cats = [1 if get_category(w, self.derogatory_dict) == "C1" else 0 for w in words]
            self.tweets.append(cleaned)
            self.category_ids.append(word_cats)
            self.labels.append(label)

    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self,idx):

        return self.tweets[idx], self.category_ids[idx], self.labels[idx]

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