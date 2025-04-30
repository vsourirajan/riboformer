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
from transformers import AutoTokenizer

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    bio_source_cols = [col for col in df.columns if 'bio_source' in col]
    y = df[bio_source_cols].to_numpy()
    return y

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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class DNATransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 78)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.mean(dim=1)
        output = self.fc(output)
        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for _, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)

            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training loss: {loss.item():.4f}, Validation loss: {val_loss:.4f}')
        print(f'Validation metrics: {val_metrics}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    y_true_mean = np.mean(all_labels, axis=1)
    y_pred_mean = np.mean(all_preds, axis=1)
    
    metrics = {
        'mse': mean_squared_error(y_true_mean, y_pred_mean),
        'r2': r2_score(y_true_mean, y_pred_mean),
        'pearson': pearsonr(y_true_mean, y_pred_mean).statistic,
        'spearman': spearmanr(y_true_mean, y_pred_mean).statistic
    }
    
    return total_loss / len(data_loader), metrics

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    data = pd.read_csv('data/final_data.csv')
    data = data.dropna()
    sequences = data['tx_sequence'].tolist()
    labels = prepare_data('data/final_data.csv')
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    dataset = DNADataset(sequences, labels, tokenizer)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'\nFold {fold+1}/10')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)
        model = DNATransformer(vocab_size=tokenizer.vocab_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        fold_metrics.append(val_metrics)
        print(f'Fold {fold+1} validation metrics: {val_metrics}')
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    print('\nAverage metrics across all folds:')
    for metric, value in avg_metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main()
