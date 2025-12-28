import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Load resources
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('models/lstm_autoencoder.keras')
    scaler = joblib.load('models/scaler.joblib')
    threshold = np.load('models/threshold.npy')[0]
    return model, scaler, threshold

def load_data():
    columns = ['unit_id', 'cycle', 'setting1', 'setting2', 'setting3', 
               's1', 's2', 's3', 's4', 's5', 's6']
    test_df = pd.read_csv('data/cmapss/test_FD001.txt', sep=' ', header=None, names=columns)
    return test_df

def main():
    st.title("LSTM Anomaly Detection Dashboard")
    st.sidebar.header("Settings")
    
    model, scaler, threshold = load_resources()
    test_df = load_data()
    
    unit_ids = test_df['unit_id'].unique()
    selected_unit = st.sidebar.selectbox("Select Unit ID", unit_ids)
    
    unit_data = test_df[test_df['unit_id'] == selected_unit].copy()
    
    # Preprocess
    features = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6']
    unit_data_scaled = unit_data.copy()
    unit_data_scaled[features] = scaler.transform(unit_data[features])
    
    # Create sequences
    seq_length = 10
    X_unit = []
    data_values = unit_data_scaled[features].values
    for i in range(len(data_values) - seq_length + 1):
        X_unit.append(data_values[i:i+seq_length])
    X_unit = np.array(X_unit)
    
    if len(X_unit) > 0:
        # Predict
        reconstructions = model.predict(X_unit)
        mse = np.mean(np.abs(reconstructions - X_unit), axis=(1, 2))
        
        # Dashboard Layout
        st.subheader(f"Sensor Data for Unit {selected_unit}")
        sensor_to_plot = st.selectbox("Select Sensor to Visualize", features[3:]) # Sensors only
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(unit_data['cycle'], unit_data[sensor_to_plot], label=sensor_to_plot)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Anomaly Detection (Reconstruction Error)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        cycles = unit_data['cycle'].values[seq_length-1:]
        ax2.plot(cycles, mse, label="Reconstruction Error (MAE)")
        ax2.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        
        # Highlight anomalies
        anomalies = mse > threshold
        ax2.scatter(cycles[anomalies], mse[anomalies], color='red', label="Anomaly")
        
        ax2.set_xlabel("Cycle")
        ax2.set_ylabel("Error")
        ax2.legend()
        st.pyplot(fig2)
        
        st.write(f"Total Cycles: {len(unit_data)}")
        st.write(f"Anomalies Detected: {np.sum(anomalies)}")
    else:
        st.write("Not enough data points for sequence length.")

if __name__ == "__main__":
    main()
