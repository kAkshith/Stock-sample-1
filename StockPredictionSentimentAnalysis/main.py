from scripts.preprocess import preprocess_data
from scripts.train_sentiment_model import train_model
from scripts.predict_sentiment import predict_sentiment
from scripts.stock_correlation import correlate_sentiment_with_stock

def get_valid_stock_symbol():
    while True:
        symbol = input("Enter stock symbol (e.g., AAPL, GOOGL): ").upper()
        # Add validation if needed
        return symbol

if __name__ == "__main__":
    # Step 1: Preprocess the data
    preprocess_data()

    # Step 2: Train the sentiment analysis model
    train_model()

    # Step 3: Get stock symbol from user
    stock_symbol = get_valid_stock_symbol()

    # Step 4: Predict sentiment
    predict_sentiment(stock_symbol)

    # Step 5: Correlate sentiment with stock performance
    correlate_sentiment_with_stock(stock_symbol)