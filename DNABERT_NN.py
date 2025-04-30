import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

def load_embeddings(path): 
    df = pd.read_pickle(path)
    return df

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    x = df[['tx_size', 'utr5_size', 'cds_size', 'utr3_size']].to_numpy()
    bio_source_cols = [col for col in df.columns if 'bio_source' in col]
    y = df[bio_source_cols].to_numpy()
    return x, y

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics for a single fold"""
    y_true_mean = np.mean(y_true, axis=1)
    y_pred_mean = np.mean(y_pred, axis=1)
    
    mse = mean_squared_error(y_true_mean, y_pred_mean)
    r2 = r2_score(y_true_mean, y_pred_mean)
    pearson = pearsonr(y_true_mean, y_pred_mean).statistic
    # print(spearmanr(y_true_mean, y_pred_mean))
    spearman = spearmanr(y_true_mean, y_pred_mean).correlation
    
    return {
        'mse': mse,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman
    }
    
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Run the DNA BERT + NN model on the embeddings and data.

        This script will load the transcript embeddings from `transcript_embeddings.pkl` and the data from
        `data/final_data.csv`, and then run the DNA BERT + NN model on the data. The script will
        split the data into 10 folds, and for each fold, it will train a model on the training set
        and evaluate it on the test set. The script will then print out the average metrics across
        the 10 folds.

        The script will print out the average metrics for the DNA BERT + NN model, including
        the mean squared error (MSE), the R-squared value, the Pearson correlation,
        and the Spearman correlation.

        :return: None
        """
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

def train_test_nn(X_train, y_train, X_test, y_test, input_dim, output_dim, epochs=50, lr=0.001):
    model = SimpleNN(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
    return y_pred

def main():
    
    embeddings_df = load_embeddings("transcript_embeddings.pkl")
    print(embeddings_df.head())

    X = embeddings_df["embedding_mean"].to_numpy()
    X = np.stack(X) 
    # currently worsens performance
    x_cols, y = prepare_data('data/final_data.csv')
    
    X = np.concatenate((X, x_cols), axis=1)
    
    X_scaler = StandardScaler().fit(X)
    X = X_scaler.transform(X)
    # X = X_scaler.fit_transform(X)
    # print(f"Min: {X.min()}, Max: {X.max()}")
    
    y_scaler = StandardScaler().fit(y)
    y = y_scaler.transform(y)
    
    #implement transformations

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    nn_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nProcessing Fold {fold}/10")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        y_pred = train_test_nn(X_train, y_train, X_test, y_test, input_dim=X.shape[1], output_dim=y.shape[1])
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)
        metrics = calculate_metrics(y_test, y_pred)
        nn_metrics.append(metrics)
    
    def average_metrics(metrics_list):
        return {
            'mse': np.mean([m['mse'] for m in metrics_list]),
            'r2': np.mean([m['r2'] for m in metrics_list]),
            'pearson': np.mean([m['pearson'] for m in metrics_list]),
            'spearman': np.mean([m['spearman'] for m in metrics_list])
        }
    
    print("\n=== Final Results (Averaged across 10 folds) ===")
    
    print("\nDNA BERT + NN:")
    nn_avg = average_metrics(nn_metrics)
    print(f"MSE: {nn_avg['mse']:.4f}")
    print(f"R2: {nn_avg['r2']:.4f}")
    print(f"Pearson Correlation: {nn_avg['pearson']:.4f}")
    print(f"Spearman Correlation: {nn_avg['spearman']:.4f}")

if __name__ == "__main__":
    main()