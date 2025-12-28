import numpy as np
import tensorflow as tf
import os

def calculate_anomaly_scores(model, data):
    reconstructions = model.predict(data)
    # Mean Absolute Error per sequence
    mse = np.mean(np.abs(reconstructions - data), axis=(1, 2))
    return mse

def find_threshold(model, train_data, percentile=99):
    mse = calculate_anomaly_scores(model, train_data)
    threshold = np.percentile(mse, percentile)
    return threshold

if __name__ == "__main__":
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    
    model = tf.keras.models.load_model('models/lstm_autoencoder.keras')
    
    threshold = find_threshold(model, X_train)
    print(f"Calculated threshold: {threshold}")
    
    test_mse = calculate_anomaly_scores(model, X_test)
    anomalies = test_mse > threshold
    print(f"Number of anomalies detected in test set: {np.sum(anomalies)} out of {len(X_test)}")
    
    np.save('models/threshold.npy', np.array([threshold]))
