import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_data(file_path):
    columns = ['unit_id', 'cycle', 'setting1', 'setting2', 'setting3', 
               's1', 's2', 's3', 's4', 's5', 's6']
    df = pd.read_csv(file_path, sep=' ', header=None, names=columns)
    return df

def preprocess_data(train_df, test_df):
    # Select features (excluding unit_id and cycle for training input)
    features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6']
    
    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    # Save scaler for later use in dashboard
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return train_df, test_df, features

def create_sequences(data, seq_length, training_mode=False):
    sequences = []
    for unit_id in data['unit_id'].unique():
        unit_data_df = data[data['unit_id'] == unit_id]
        
        # If training, only use the first 80% of data (assumed healthy)
        if training_mode:
            cutoff = int(len(unit_data_df) * 0.80)
            unit_data = unit_data_df.iloc[:cutoff, 2:].values
        else:
            unit_data = unit_data_df.iloc[:, 2:].values
            
        if len(unit_data) >= seq_length:
            for i in range(len(unit_data) - seq_length + 1):
                sequences.append(unit_data[i:i+seq_length])
    return np.array(sequences)

if __name__ == "__main__":
    train_df = load_data('data/cmapss/train_FD001.txt')
    test_df = load_data('data/cmapss/test_FD001.txt')
    
    train_df, test_df, features = preprocess_data(train_df, test_df)
    
    SEQ_LENGTH = 10
    # Enable training_mode=True for X_train to learn only from healthy data
    X_train = create_sequences(train_df, SEQ_LENGTH, training_mode=True)
    X_test = create_sequences(test_df, SEQ_LENGTH, training_mode=False)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
