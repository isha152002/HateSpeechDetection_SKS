import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for glove, categories, labels in test_loader:
            # YOUR CODE:
            hate_pred,_= model(glove,categories)
            preds = (hate_pred.squeeze(1) >= 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    
    # compute accuracy and f1
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1
    