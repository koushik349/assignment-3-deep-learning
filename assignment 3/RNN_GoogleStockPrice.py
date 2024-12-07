import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

df = pd.read_csv("Google_Stock_Price_Train.csv")

# Displaying the first few rows of the dataframe to check the data
print("Initial data preview:")
print(df.head())

# Step 3: Converting 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Verifying that the Date column has been converted successfully
print("\nData after converting 'Date' column to datetime:")
print(df.head())

# Step 4: Handling missing values
print("\nChecking for missing values in the dataset:")
print(df.isnull().sum())

# Handling missing values by forward filling them
df.bfill(inplace=True)

# Verifying that missing values have been handled
print("\nMissing values after forward filling:")
print(df.isnull().sum())

# Step 5: Converting numerical columns to numeric types
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

# Checking the data types of the columns to ensure they are now numeric
print("\nData types of the columns after conversion:")
print(df.dtypes)

# Displaying the cleaned dataframe
print("\nCleaned data preview:")
print(df.head())

# Step 8: Saveing the cleaned data to a new CSV file (optional)

df.to_csv("Cleaned_Google_Stock_Price_Train.csv", index=False)

print("\nCleaned data has been saved as 'Cleaned_Google_Stock_Price_Train.csv'.")

# Step 2: Loading the cleaned dataset
df = pd.read_csv("Cleaned_Google_Stock_Price_Train.csv")

# Step 3: Selecting the 'Close' column for prediction
df_close = df[['Date', 'Close']]

# Step 4: Normalizing the 'Close' prices , Using Min-Max scaling to normalize the data to a range [0, 1].
scaler = MinMaxScaler(feature_range=(0, 1))
df_close.loc[:, 'Close'] = scaler.fit_transform(df_close[['Close']])

# Step 5: Creating sequences for RNN training,  where the length of each sequence is 'look_back' days.
# 'look_back' represents the number of previous days used to predict the next day's price.
look_back = 60  # 60 days of stock prices to predict the 61st day.

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])  # Previous 'look_back' days
        y.append(data[i, 0])  # The price of the next day
    return np.array(X), np.array(y)

# Converting the 'Close' column to a numpy array for sequence generation
close_prices = df_close['Close'].values
close_prices = close_prices.reshape(-1, 1)

# Creating sequences
X, y = create_sequences(close_prices, look_back)

# Step 6: Reshaping the input data for RNN
# The RNN expects data in the shape of [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 7: Spliting the data into training and testing sets , Typically, 80% of the data is used for training and 20% for testing.
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 8: Verifying the shapes of the training and testing sets
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Step 9: Saving the preprocessed data

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("\nData preparation is complete. The processed data has been saved.")




def load_and_clean_data(file_path):
   
    try:
        # Loading the dataset
        df = pd.read_csv(file_path)
        
        # Converting 'Close' column to numeric and remove commas
        df['Close'] = df['Close'].str.replace(',', '').astype(float)
        
        # Selecting the 'Date' and 'Close' columns for prediction
        df = df[['Date', 'Close']]
        
        # Converting 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def normalize_data(df):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler



def prepare_data(scaled_data, look_back=60):
    """
    Prepare training and testing datasets
    
    Args:
        scaled_data (np.array): Normalized stock price data
        look_back (int): Number of previous days to use for prediction
    
    Returns:
        tuple: X_train, y_train, X_test, y_test
    """
    # Spliting the data into training and testing sets (80% training, 20% testing)
    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len:]

    # Creating X_train and y_train
    X_train, y_train = [], []
    for i in range(look_back, len(train_data)):
        X_train.append(train_data[i-look_back:i, 0])
        y_train.append(train_data[i, 0])

    # Creating X_test and y_test
    X_test, y_test = [], []
    for i in range(look_back, len(test_data)):
        X_test.append(test_data[i-look_back:i, 0])
        y_test.append(test_data[i, 0])

    # Converting to numpy arrays and reshape
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test



def build_rnn_model(input_shape):
    """
    Build LSTM RNN model for stock price prediction
    
    Args:
        input_shape (tuple): Shape of input data
    
    Returns:
        tensorflow.keras.Model: Compiled RNN model
    """
    model = Sequential([
        Input(shape=input_shape),  # Specify the input shape using the Input layer
        LSTM(units=50, return_sequences=True),
        Dropout(0.3),
        LSTM(units=50, return_sequences=False),
        Dropout(0.3),
        Dense(units=1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model




def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the RNN model
    
    Args:
        model (tensorflow.keras.Model): RNN model
        X_train, y_train, X_test, y_test: Training and testing datasets
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        History of model training
    """
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    return history



def calculate_metrics(true_prices, predicted_prices):
        metrics = {
        'Mean Squared Error (MSE)': mean_squared_error(true_prices, predicted_prices),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(true_prices, predicted_prices)),
        'Mean Absolute Error (MAE)': mean_absolute_error(true_prices, predicted_prices),
        'R-Squared (R²)': r2_score(true_prices, predicted_prices),
        'Mean Absolute Percentage Error (MAPE)': mean_absolute_percentage_error(true_prices, predicted_prices) * 100
    }
    return metrics



def plot_results(true_prices, predicted_prices):
    """
    Plot actual vs predicted stock prices
    
    """
    plt.figure(figsize=(12, 6))
    plt.plot(true_prices, color='blue', label='Actual Stock Price')
    plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    
def main():
   
    file_path = "Google_Stock_Price_Train.csv"
    
    # Loading and preprocessing data
    df = load_and_clean_data(file_path)
    if df is None:
        return
    
    # Normalizing  data
    scaled_data, scaler = normalize_data(df)
    
    # Preparing data
    X_train, y_train, X_test, y_test = prepare_data(scaled_data)
    
    # Building and training model
    model = build_rnn_model(input_shape=(X_train.shape[1], 1))
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Predicting prices
    predicted_scaled = model.predict(X_test)
    
    # Inversing transform to get actual prices
    predicted_prices = scaler.inverse_transform(predicted_scaled)
    true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculating and print metrics
    metrics = calculate_metrics(true_prices, predicted_prices)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    
    # Ploting results
    plot_results(true_prices, predicted_prices)

if __name__ == "__main__":
    main()