import torch
from torch.optim import RMSprop
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
from config import LEARNING_RATE, EPOCHS, PATIENCE
from training.loss import combined_loss

def train(model, hate_loader, sentiment_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) #move model to GPU
    optimizer = RMSprop(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = 0.0  # higher is better for F1
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()  # set model to training mode
        
        # alternate between hate speech and sentiment batches
        for (hate_batch, sentiment_batch) in zip(hate_loader, sentiment_loader):
            # unpack hate batch
            tweets, word_category_ids, hate_labels = hate_batch
            hate_labels = hate_labels.to(device)
            
            # unpack sentiment batch
            tweets_s, word_category_ids_s, sentiment_labels = sentiment_batch
            sentiment_labels = sentiment_labels.to(device)
            
            optimizer.zero_grad()
            hate_pred, _=model(tweets, word_category_ids)
            _, sentiment_pred =model(tweets_s, word_category_ids_s)
            loss = combined_loss(hate_pred, hate_labels, sentiment_pred, sentiment_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
    
            for tweets_v, word_category_ids_v, labels in val_loader:
                
                labels = labels.to(device)
        
                hate_pred, _ = model(tweets_v, word_category_ids_v)
                preds = (hate_pred.squeeze(1) >= 0.5).long()
        
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            print(f'Epoch {epoch+1}/{EPOCHS} — Val F1: {val_f1:.4f}')
    
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f'Early stopping at epoch {epoch+1}')
                    break