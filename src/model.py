import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout

def build_lstm_autoencoder(seq_length, num_features):
    model = Sequential([
        # Encoder
        LSTM(64, activation='relu', input_shape=(seq_length, num_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        RepeatVector(seq_length),
        # Decoder
        LSTM(32, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

if __name__ == "__main__":
    # Test model build
    model = build_lstm_autoencoder(10, 9)
    model.summary()
