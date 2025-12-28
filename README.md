# LSTM-based Anomaly Detection for Time Series Data

## Project Overview
This project implements a deep learning anomaly detection system for multivariate time series sensor data (simulating NASA Turbofan jet engines). It uses an **LSTM Autoencoder** to learn the patterns of healthy engine cycles and detects anomalies based on reconstruction error.

## Key Features
*   **Synthetic Data Generation**: Custom script to generate realistic sensor data with exponential degradation trends.
*   **Deep Learning Model**: LSTM Autoencoder with Dropout layers for robust unsupervised learning.
*   **Intelligent Preprocessing**: Training strategy focused on the "healthy" phase of engine life to maximize anomaly sensitivity.
*   **Interactive Dashboard**: Streamlit-based UI for real-time visualization of sensor data and anomaly detection.

## Performance
After optimization, the model achieves high reliability:
*   **Precision**: **82.60%** (Low false alarm rate)
*   **Recall**: **99.64%** (Detects almost all failures)
*   **F1 Score**: **90.32%**

## Project Structure
```
E:\Task\
├── data/               # Generated datasets
├── models/             # Saved Keras model and scalers
├── notebooks/          # Jupyter notebook for analysis
├── src/                # Source code
│   ├── anomaly_detector.py # Threshold calculation & detection logic
│   ├── dashboard.py        # Streamlit dashboard application
│   ├── evaluate.py         # Performance evaluation script
│   ├── generate_data.py    # Synthetic data generator
│   ├── model.py            # LSTM Autoencoder architecture
│   ├── preprocess.py       # Data normalization & sequencing
│   └── train.py            # Model training script
├── IMPROVEMENTS.md     # Log of optimization steps
├── REPORT.md           # Detailed technical report
└── README.md           # This file
```

## Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy tensorflow streamlit scikit-learn matplotlib joblib
    ```

2.  **Generate Data**:
    ```bash
    python src/generate_data.py
    ```

3.  **Run Training Pipeline**:
    ```bash
    python src/preprocess.py
    python src/train.py
    ```

## Usage

### Run Dashboard
To visualize the results interactively:
```bash
python -m streamlit run src/dashboard.py
```

### Run Evaluation
To calculate Precision, Recall, and F1 Score:
```bash
python src/evaluate.py
```
