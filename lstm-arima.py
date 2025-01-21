import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.arima.model import ARIMA
import math
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('BHARTIARTL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('B')
df = df.fillna(method='ffill')  # Handle missing values
data = df['Close'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model with dropout
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile and train with early stopping
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and process residuals
train_residuals = y_train_inv - train_predict
test_residuals = y_test_inv - test_predict

try:
    # Fit ARIMA on residuals with a simpler model
    train_residuals_series = pd.Series(train_residuals.flatten())
    train_residuals_series = train_residuals_series.fillna(method='ffill')  # Handle any NaN values
    
    arima_model = ARIMA(train_residuals_series, order=(1,1,1))  # Simpler ARIMA model
    arima_model_fit = arima_model.fit()
    
    # Forecast residuals
    residual_forecast = arima_model_fit.forecast(steps=len(test_residuals))
    residual_forecast = residual_forecast.values.reshape(-1, 1)
    
    # Combine predictions
    combined_train = train_predict + arima_model_fit.fittedvalues.values.reshape(-1, 1)
    combined_test = test_predict + residual_forecast
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train_inv, combined_train)
    train_rmse = np.sqrt(train_mse)
    test_mse = mean_squared_error(y_test_inv, combined_test)
    test_rmse = np.sqrt(test_mse)
    
    print('\nModel Performance Metrics:')
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')
    
    # Plot results
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'], label='Actual Price', alpha=0.7)
    plt.plot(df.index[time_step:len(combined_train)+time_step], 
             combined_train, label='Train Predictions', alpha=0.7)
    plt.plot(df.index[len(combined_train)+time_step*2+1:len(df)-1], 
             combined_test, label='Test Predictions', alpha=0.7)
    plt.title('Stock Price Prediction - Hybrid LSTM-ARIMA Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

except Exception as e:
    print(f"Error in ARIMA modeling: {str(e)}")
    # Fall back to LSTM predictions only
    print("\nFalling back to LSTM predictions only...")
    
    train_mse = mean_squared_error(y_train_inv, train_predict)
    train_rmse = np.sqrt(train_mse)
    test_mse = mean_squared_error(y_test_inv, test_predict)
    test_rmse = np.sqrt(test_mse)
    
    print('\nLSTM Model Performance Metrics:')
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')