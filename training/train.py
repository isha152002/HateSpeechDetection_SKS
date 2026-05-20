import torch
from torch.optim import RMSprop
from config import LEARNING_RATE, EPOCHS, PATIENCE
from training.loss import combined_loss

def train(model, hate_loader, sentiment_loader, val_loader):
    optimizer = RMSprop(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')  # infinity — any loss will be lower
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()  # set model to training mode
        
        # alternate between hate speech and sentiment batches
        for (hate_batch, sentiment_batch) in zip(hate_loader, sentiment_loader):
            # unpack hate batch
            glove, categories, hate_labels = hate_batch
            
            # unpack sentiment batch
            glove_s, categories_s, sentiment_labels = sentiment_batch
            
            optimizer.zero_grad()
            hate_pred, _=model(glove,categories)
            _, sentiment_pred =model(glove_s,categories_s)
            loss = combined_loss(hate_pred, hate_labels, sentiment_pred, sentiment_labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():  # don't compute gradients during validation
            # validation loop — compute val loss
            val_loss=0
            for glove, categories, labels in val_loader:
                hate_pred, _ = model(glove, categories)
                # compute val loss — only hate speech task on validation
                val_loss += combined_loss(hate_pred, labels, hate_pred, labels).item()

            print(f'Epoch {epoch+1}/{EPOCHS} — Val Loss: {val_loss:.4f}')
            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'checkpoints/best_model.pt')  # save best model
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f'Early stopping at epoch {epoch}')
                    break