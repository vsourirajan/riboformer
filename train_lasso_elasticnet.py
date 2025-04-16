import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd

def encode_sequence(seq, max_len):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    encoded = [mapping.get(nuc, [0,0,0,0]) for nuc in seq[:max_len]]
    while len(encoded) < max_len:
        encoded.append([0,0,0,0])
    return np.array(encoded).flatten()

def prepare_data(data_path):
    #load the data  
    df = pd.read_csv(data_path)

    #encode the sequence
    X_numeric = df[['utr5_size', 'cds_size', 'utr3_size']].to_numpy()
    sequences = df['tx_sequence'].tolist()
    #X_seq = np.array([encode_sequence(seq, max_len=33681) for seq in sequences])
    #X = np.hstack([X_numeric, X_seq])

    bio_source_cols = [col for col in df.columns if 'bio_source' in col]
    y = df[bio_source_cols].to_numpy()

    return X_numeric, y

def main():
    X, y = prepare_data('data/CLEANED_data_with_human_TE_cellline_all_plain.csv')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"MSE_LASSO: {mse}")
    print(f"R2_LASSO: {r2_score(y_val, y_pred)}")


    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X_train, y_train)
    y_pred = elastic_net.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"MSE_ELASTIC_NET: {mse}")
    print(f"R2_ELASTIC_NET: {r2_score(y_val, y_pred)}")

if __name__ == "__main__":
    main()