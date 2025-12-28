import numpy as np
import tensorflow as tf
from model import build_lstm_autoencoder
import os

def train():
    X_train = np.load('data/X_train.npy')
    
    seq_length = X_train.shape[1]
    num_features = X_train.shape[2]
    
    model = build_lstm_autoencoder(seq_length, num_features)
    
    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    
    history = model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_autoencoder.keras')
    print("Model saved to models/lstm_autoencoder.keras")
    
    return history

if __name__ == "__main__":
    train()
