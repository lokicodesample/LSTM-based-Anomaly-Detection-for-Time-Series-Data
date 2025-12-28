# LSTM-based Anomaly Detection Report

## 1. Objective
Implement a deep learning anomaly detection system for multivariate time series sensor data using an LSTM Autoencoder. The goal is to accurately detect impending engine failures while minimizing false alarms.

## 2. Methodology

### Data Acquisition & Generation
*   **Source**: Synthetic data mimicking the NASA CMAPSS dataset structure.
*   **Augmentation**: Generated data for **100 engine units** (up from 20) to improve model generalization.
*   **Degradation**: Implemented **exponential growth patterns** on specific sensors (s2, s4, s6) to simulate realistic wear and tear towards the end of an engine's lifespan.

### Preprocessing
*   **Normalization**: Used `MinMaxScaler` to scale all sensor values between 0 and 1.
*   **Sequencing**: Created sliding window sequences of length 10.
*   **Training Strategy**: 
    *   **Healthy Data Split**: The model was trained **only on the first 80%** of each engine's lifespan. 
    *   **Reasoning**: This ensures the Autoencoder learns strictly "normal" operational patterns. When the engine enters the final 20% (failure phase), the reconstruction error spikes significantly, allowing for easy detection.

### Model Architecture
*   **Type**: LSTM Autoencoder (Unsupervised Learning).
*   **Layers**:
    *   **Encoder**: LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2)
    *   **Bottleneck**: RepeatVector(seq_length)
    *   **Decoder**: LSTM(32) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2)
    *   **Output**: TimeDistributed(Dense)
*   **Optimization**: Added **Dropout layers** to prevent overfitting, resulting in much better generalization on unseen test data.

### Anomaly Detection
*   **Metric**: Mean Absolute Error (MAE) of the reconstruction.
*   **Thresholding**: Calculated at the **99th percentile** of the training reconstruction error. This conservative threshold significantly reduced false positives (high precision).

## 3. Final Performance Evaluation
The model was evaluated on a separate test set containing 20 engine units.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Precision** | **82.60%** | High confidence in alarms. Very few false positives. |
| **Recall** | **99.64%** | Excellent safety profile. Virtually all failures are detected. |
| **F1 Score** | **90.32%** | Robust overall performance. |

## 4. Deliverables
*   **Source Code**: `src/` directory containing all Python scripts.
*   **Dashboard**: A Streamlit application (`src/dashboard.py`) for real-time visualization.
*   **Notebook**: `notebooks/Anomaly_Detection.ipynb` demonstrating the workflow.
*   **Documentation**: `README.md` and this Report.

## 5. How to Run
1.  **Generate Data**: `python src/generate_data.py`
2.  **Preprocess**: `python src/preprocess.py`
3.  **Train**: `python src/train.py`
4.  **Detect & Evaluate**: `python src/anomaly_detector.py` and `python src/evaluate.py`
5.  **Visualize**: `python -m streamlit run src/dashboard.py`