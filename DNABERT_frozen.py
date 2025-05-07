import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os
import json
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')
import math

class DNABERTFrozenRegressor(nn.Module):
    def __init__(self, input_dim=768, output_dim=78, bert_model_name='zhihan1996/DNA_bert_6', hidden_dim=256, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.rnn = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2 * hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        rnn_out, _ = self.rnn(embeddings)
        pooled = rnn_out.mean(dim=1)
        rnn_out = (rnn_out - rnn_out.mean(dim=1, keepdim=True)) / (rnn_out.std(dim=1, keepdim=True) + 1e-6)
        return self.regressor(pooled).squeeze(1)

class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    bio_source_cols = [col for col in df.columns if 'bio_source' in col]
    return df['tx_sequence'].tolist(), df[bio_source_cols].to_numpy()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for _, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}, Metrics = {val_metrics}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'{name}: grad mean =', param.grad.abs().mean().item() if param.grad is not None else 'None')


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    y_true = np.mean(all_labels, axis=1)
    y_pred = np.mean(all_preds, axis=1)   
    print(np.unique(y_pred))
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson': pearsonr(y_true, y_pred).statistic,
        'spearman': spearmanr(y_true, y_pred).correlation
    }
    return total_loss / len(data_loader), metrics

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')    
    sequences, labels = prepare_data('data/final_data.csv')
    tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    dataset = DNADataset(sequences, labels, tokenizer)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'\nFold {fold+1}/10')
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False)
        model = DNABERTFrozenRegressor().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        fold_metrics.append(val_metrics)
        print(f'Fold {fold+1} validation metrics: {val_metrics}')
    avg_metrics = {k: np.mean([fold[k] for fold in fold_metrics]) for k in fold_metrics[0]}
    print('\nAverage metrics across all folds:')
    for k, v in avg_metrics.items():
        print(f'{k}: {v:.4f}')

if __name__ == '__main__':
    main()