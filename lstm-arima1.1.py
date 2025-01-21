import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import ta

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Define parameters
time_step = 60
batch_size = 32
epochs = 50
validation_split = 0.1
train_split = 0.8

# Load and preprocess data
df = pd.read_csv('AXISBANK.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df['Close'].values.reshape(-1, 1)

# Technical indicators function
def add_features(df):
    # Technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd_diff()
    
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    
    # Fill NaN values
    return df.ffill()

df = add_features(df)

# Select features
feature_columns = ['Close', 'RSI', 'MACD', 'MA5', 'MA20', 'BB_upper', 'BB_lower']

# Separate scalers for features and target
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Scale features and target separately
scaled_features = feature_scaler.fit_transform(df[feature_columns])
scaled_target = target_scaler.fit_transform(df[['Close']])

# Split data
train_size = int(len(scaled_features) * train_split)
train_features = scaled_features[:train_size]
test_features = scaled_features[train_size:]
train_target = scaled_target[:train_size]
test_target = scaled_target[train_size:]

# Create sequences
def create_sequences(features, target, time_step):
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        X.append(features[i:(i + time_step)])
        y.append(target[i + time_step])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_features, train_target, time_step)
X_test, y_test = create_sequences(test_features, test_target, time_step)

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(feature_columns))
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(feature_columns))

# Enhanced model architecture
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(time_step, len(feature_columns))),
    BatchNormalization(),
    Dropout(0.4),
    
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    
    Bidirectional(LSTM(32)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dense(1)
])

# Model compilation with fixed learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='huber')

# Callbacks with learning rate reduction
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
]

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_split=validation_split,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and actual values
train_predict = target_scaler.inverse_transform(train_predict)
test_predict = target_scaler.inverse_transform(test_predict)
y_train_inv = target_scaler.inverse_transform(y_train)
y_test_inv = target_scaler.inverse_transform(y_test)

# Calculate residuals
train_residuals = y_train_inv - train_predict
test_residuals = y_test_inv - test_predict

# ARIMA on residuals
train_residuals_series = pd.Series(train_residuals.flatten())
arima_model = ARIMA(train_residuals_series, order=(1,1,1))
arima_model_fit = arima_model.fit()

# Forecast residuals
residual_forecast = arima_model_fit.forecast(steps=len(test_residuals))
residual_forecast = residual_forecast.values.reshape(-1, 1)

# Combine predictions
combined_train = train_predict + arima_model_fit.fittedvalues.values.reshape(-1, 1)
combined_test = test_predict + residual_forecast

def directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy of predictions"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate direction of movement
    y_true_dir = np.diff(y_true) > 0
    y_pred_dir = np.diff(y_pred) > 0
    
    # Compare directions
    return np.mean(y_true_dir == y_pred_dir) * 100

def calculate_performance_metrics(y_true, y_pred, set_name=""):
    """Calculate comprehensive performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    da = directional_accuracy(y_true, y_pred)
    
    print(f"\n{set_name} Performance Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Directional Accuracy: {da:.2f}%")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'da': da}

# Calculate metrics
train_metrics = calculate_performance_metrics(y_train_inv, combined_train, "Training")
test_metrics = calculate_performance_metrics(y_test_inv, combined_test, "Testing")

# Visualize results
plt.figure(figsize=(15, 10))

# Plot 1: Training History
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot 2: Predictions vs Actual
plt.subplot(2, 2, 2)
plt.scatter(y_test_inv, combined_test, alpha=0.5)
plt.plot([y_test_inv.min(), y_test_inv.max()], 
         [y_test_inv.min(), y_test_inv.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Prediction vs Actual')

# Plot 3: Time Series Plot
plt.subplot(2, 1, 2)
plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual', alpha=0.7)
plt.plot(df.index[-len(combined_test):], combined_test, label='Predicted', alpha=0.7)
plt.title('Time Series Comparison')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()

# Save metrics to file
metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE', 'DA'],
    'Training': [train_metrics[k] for k in ['rmse', 'mae', 'r2', 'mape', 'da']],
    'Testing': [test_metrics[k] for k in ['rmse', 'mae', 'r2', 'mape', 'da']]
})
metrics_df.to_csv('model_metrics.csv', index=False)

# Calculate directional accuracy
def directional_accuracy(y_true, y_pred):
    y_true_direction = np.sign(np.diff(y_true.flatten()))
    y_pred_direction = np.sign(np.diff(y_pred.flatten()))
    return np.mean(y_true_direction == y_pred_direction) * 100

da_train = directional_accuracy(y_train_inv, combined_train)
da_test = directional_accuracy(y_test_inv, combined_test)
print(f"\nDirectional Accuracy:")
print(f"Training: {da_train:.2f}%")
print(f"Testing: {da_test:.2f}%")