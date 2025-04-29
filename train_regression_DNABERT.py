import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
    df_filtered = df[df['tx_sequence'].apply(len).between(500, 1500)].reset_index(drop=True)
    bio_source_cols = [col for col in df_filtered.columns if 'bio_source' in col]
    y = df_filtered[bio_source_cols].to_numpy()

    return y

def main():
    embeddings_df = load_embeddings("transcript_embeddings.pkl")
    print(embeddings_df.head())

    X = embeddings_df["embedding_mean"].to_numpy()
    X = np.stack(X) 
    y = prepare_data('data/CLEANED_data_with_human_TE_cellline_all_plain.csv')

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"MSE_LINEAR_REGRESSION: {mse}")
    print(f"R2_LINEAR_REGRESSION: {r2_score(y_val, y_pred)}")
    # print("PEARSON CORRELATION: ", pearsonr(y_val, y_pred))
    # print("SPEARMAN CORRELATION: ", spearmanr(y_val, y_pred))


    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"MSE_LASSO: {mse}")
    print(f"R2_LASSO: {r2_score(y_val, y_pred)}")
    # print("PEARSON CORRELATION: ", pearsonr(y_val, y_pred))
    # print("SPEARMAN CORRELATION: ", spearmanr(y_val, y_pred))


    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X_train, y_train)
    y_pred = elastic_net.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"MSE_ELASTIC_NET: {mse}")
    print(f"R2_ELASTIC_NET: {r2_score(y_val, y_pred)}")
    # print("PEARSON CORRELATION: ", pearsonr(y_val, y_pred))
    # print("SPEARMAN CORRELATION: ", spearmanr(y_val, y_pred))

if __name__ == "__main__":
    main()
    