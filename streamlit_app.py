import streamlit as st

st.title('Trial app ')

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Streamlit App Title
st.title("Stock Price Prediction using LSTM")

# User Input for Stock Symbol
stock_symbol = st.text_input("Enter Stock Ticker:", "AAPL")

# Fetch Stock Data
def load_data(stock_symbol):
    df = yf.download(stock_symbol, period="5y")
    return df

data = load_data(stock_symbol)
st.write("Stock Data", data.tail())

# Preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

scaled_data, scaler = preprocess_data(data)

# Create training data
def create_train_data(scaled_data, time_step=60):
    x_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        x_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    return np.array(x_train), np.array(y_train)

x_train, y_train = create_train_data(scaled_data)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM Model
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model()
st.write("LSTM Model Summary")
st.text(model.summary())

# Train the model
if st.button("Train Model"):
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    st.write("Model Training Completed!")

# Prediction
def predict_future(model, data, scaler, time_step=60):
    inputs = data['Close'].values[-time_step:].reshape(-1, 1)
    inputs = scaler.transform(inputs)
    inputs = np.reshape(inputs, (1, inputs.shape[0], 1))
    predicted_price = model.predict(inputs)
    return scaler.inverse_transform(predicted_price)[0][0]

if st.button("Predict Next Day Price"):
    prediction = predict_future(model, data, scaler)
    st.write(f"Predicted Next Day Closing Price: ${prediction:.2f}")

# Plot Stock Prices
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Actual Price')
plt.legend()
st.pyplot(fig)
