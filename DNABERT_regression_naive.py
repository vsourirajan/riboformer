import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def load_embeddings(path): 
    df = pd.read_pickle(path)
    return df

def prepare_data(data_path):
    #load the data  
    df = pd.read_csv(data_path)
    bio_source_cols = [col for col in df.columns if 'bio_source' in col]
    y = df[bio_source_cols].to_numpy()
    return y

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics for a single fold"""
    y_true_mean = np.mean(y_true, axis=1)
    y_pred_mean = np.mean(y_pred, axis=1)
    
    mse = mean_squared_error(y_true_mean, y_pred_mean)
    r2 = r2_score(y_true_mean, y_pred_mean)
    pearson = pearsonr(y_true_mean, y_pred_mean).statistic
    spearman = spearmanr(y_true_mean, y_pred_mean).statistic
    
    return {
        'mse': mse,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman
    }

def main():
    embeddings_df = load_embeddings("transcript_embeddings.pkl")
    print(embeddings_df.head())

    X = embeddings_df["embedding_mean"].to_numpy()
    X = np.stack(X) 
    y = prepare_data('data/final_data.csv')

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Initialize lists to store metrics for each fold
    linear_metrics = []
    lasso_metrics = []
    elastic_net_metrics = []
    
    # Perform k-fold cross validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nProcessing Fold {fold}/10")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Linear Regression
        linear_regression = LinearRegression()
        linear_regression.fit(X_train, y_train)
        y_pred = linear_regression.predict(X_test)
        linear_metrics.append(calculate_metrics(y_test, y_pred))
        
        # Lasso Regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        lasso_metrics.append(calculate_metrics(y_test, y_pred))
        
        # Elastic Net Regression
        elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elastic_net.fit(X_train, y_train)
        y_pred = elastic_net.predict(X_test)
        elastic_net_metrics.append(calculate_metrics(y_test, y_pred))
    
    # Calculate average metrics
    def average_metrics(metrics_list):
        return {
            'mse': np.mean([m['mse'] for m in metrics_list]),
            'r2': np.mean([m['r2'] for m in metrics_list]),
            'pearson': np.mean([m['pearson'] for m in metrics_list]),
            'spearman': np.mean([m['spearman'] for m in metrics_list])
        }
    
    # Print results
    print("\n=== Final Results (Averaged across 10 folds) ===")
    
    print("\nLinear Regression:")
    linear_avg = average_metrics(linear_metrics)
    print(f"MSE: {linear_avg['mse']:.4f}")
    print(f"R2: {linear_avg['r2']:.4f}")
    print(f"Pearson Correlation: {linear_avg['pearson']:.4f}")
    print(f"Spearman Correlation: {linear_avg['spearman']:.4f}")
    
    print("\nLasso Regression:")
    lasso_avg = average_metrics(lasso_metrics)
    print(f"MSE: {lasso_avg['mse']:.4f}")
    print(f"R2: {lasso_avg['r2']:.4f}")
    print(f"Pearson Correlation: {lasso_avg['pearson']:.4f}")
    print(f"Spearman Correlation: {lasso_avg['spearman']:.4f}")
    
    print("\nElastic Net Regression:")
    elastic_net_avg = average_metrics(elastic_net_metrics)
    print(f"MSE: {elastic_net_avg['mse']:.4f}")
    print(f"R2: {elastic_net_avg['r2']:.4f}")
    print(f"Pearson Correlation: {elastic_net_avg['pearson']:.4f}")
    print(f"Spearman Correlation: {elastic_net_avg['spearman']:.4f}")

if __name__ == "__main__":
    main()
    