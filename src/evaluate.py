import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf

def evaluate():
    X_test = np.load('data/X_test.npy')
    threshold = np.load('models/threshold.npy')[0]
    model = tf.keras.models.load_model('models/lstm_autoencoder.keras')
    
    reconstructions = model.predict(X_test)
    mse = np.mean(np.abs(reconstructions - X_test), axis=(1, 2))
    predictions = mse > threshold
    
    # Define ground truth: For synthetic data, let's say the last 20% of cycles for each unit are anomalies
    # We need to map sequences back to their units to do this properly, 
    # but for a simple evaluation, we can estimate.
    
    # Load original test data to get unit info
    columns = ['unit_id', 'cycle', 'setting1', 'setting2', 'setting3', 
               's1', 's2', 's3', 's4', 's5', 's6']
    test_df = pd.read_csv('data/cmapss/test_FD001.txt', sep=' ', header=None, names=columns)
    
    seq_length = 10
    gt_anomalies = []
    
    for unit_id in test_df['unit_id'].unique():
        unit_data = test_df[test_df['unit_id'] == unit_id]
        lifespan = len(unit_data)
        # Mark last 15% as anomalous
        anomaly_start = int(lifespan * 0.85)
        
        unit_gt = [1 if i >= anomaly_start else 0 for i in range(lifespan)]
        # sequences start from 0 to lifespan-seq_length
        if lifespan >= seq_length:
            for i in range(lifespan - seq_length + 1):
                # sequence is anomalous if its last point is in the anomaly zone
                gt_anomalies.append(unit_gt[i + seq_length - 1])
                
    gt_anomalies = np.array(gt_anomalies)
    
    # Align lengths if necessary (should match if logic is correct)
    min_len = min(len(predictions), len(gt_anomalies))
    predictions = predictions[:min_len]
    gt_anomalies = gt_anomalies[:min_len]
    
    precision = precision_score(gt_anomalies, predictions)
    recall = recall_score(gt_anomalies, predictions)
    f1 = f1_score(gt_anomalies, predictions)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    evaluate()
