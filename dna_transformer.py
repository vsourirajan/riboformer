# Full script to train a simple DNA Transformer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import random

# ------------- Tokenizer -------------
token_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'U': 4}

def tokenize(seq):
    return [token_dict.get(c, 0) for c in seq]

# ------------- Dataset Class -------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(tokenize(self.sequences[idx]), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label

# ------------- Collate Function (for padding) -------------
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences, labels

# ------------- Transformer Model -------------
class SimpleDNATransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# ------------- Training Loop -------------
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df_filtered = df[df['tx_sequence'].apply(len).between(500, 1500)].reset_index(drop=True)
    bio_source_cols = [col for col in df_filtered.columns if 'bio_source' in col]
    sequences = df_filtered['tx_sequence'].tolist()
    labels = df_filtered[bio_source_cols].to_numpy()
    return sequences, labels

if __name__ == "__main__":
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    output_dim = 78
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3
    data_path = "data/CLEANED_data_with_human_TE_cellline_all_plain.csv"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load Data
    sequences, labels = prepare_data(data_path)
    dataset = SequenceDataset(sequences, labels)

    # Train/Test Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = SimpleDNATransformer(vocab_size=5, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, output_dim=output_dim)
    model = model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "dna_transformer.pth")
    print("Training complete and model saved.")
