from utils.data_loader import  HateSpeechDataset, SentimentDataset, create_dataloader
from utils.dictionary import load_dictionary
from model.sks_model import SksModel
from training.train import train
from evaluation.evaluate import evaluate
from torch.utils.data import ConcatDataset
import torch


DICT_PATH       = 'data/raw/derogatory_dictionary.csv'
DAVIDSON_PATH   = 'data/raw/davidson.csv'
SEMEVAL_PATH    = 'data/raw/semeval_train.parquet'
SEMEVAL_VAL     = 'data/raw/semeval_val.parquet'
SEMEVAL_TEST    = 'data/raw/semeval_test.parquet'
SENTIMENT_PATH  = 'data/raw/kaggle_sentiment.csv'


dictionary=load_dictionary(DICT_PATH )
davidson_ds=HateSpeechDataset(DAVIDSON_PATH, dictionary,'davidson')
semeval_train_ds=HateSpeechDataset(SEMEVAL_PATH, dictionary,'semeval')
semeval_val_ds=HateSpeechDataset(SEMEVAL_VAL, dictionary,'semeval')
semeval_test_ds=HateSpeechDataset(SEMEVAL_TEST, dictionary,'semeval')
sentiment_ds=SentimentDataset(SENTIMENT_PATH, dictionary)


combined_hate_ds = ConcatDataset([semeval_train_ds, davidson_ds])
combined_hate_loader = create_dataloader(combined_hate_ds, shuffle=True)

davidson_loader = create_dataloader(davidson_ds, shuffle=True)
semeval_loader = create_dataloader(semeval_train_ds, shuffle=True)
semeval_val_loader = create_dataloader(semeval_val_ds, shuffle=False)
semeval_test_loader = create_dataloader(semeval_test_ds, shuffle=False)
sentiment_loader = create_dataloader(sentiment_ds, shuffle=True)

sks=SksModel()
train(sks, combined_hate_loader, sentiment_loader, semeval_val_loader)
sks.load_state_dict(torch.load('checkpoints/best_model.pt'))

accuracy, f1 = evaluate(sks, semeval_test_loader)
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')

# evaluate on davidson as well
accuracy_dv, f1_dv = evaluate(sks, davidson_loader)
print(f'Davidson — Accuracy: {accuracy_dv:.4f}')
print(f'Davidson — F1 Score: {f1_dv:.4f}')