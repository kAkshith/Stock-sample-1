
from scripts.preprocess import preprocess_data
from scripts.train_sentiment_model import train_model
from scripts.predict_sentiment import predict_sentiment
from scripts.stock_correlation import correlate_sentiment_with_stock

if __name__ == "__main__":
    # Step 1: Preprocess the data
    preprocess_data()

    # Step 2: Train the sentiment analysis model
    train_model()

    # Step 3: Predict sentiment
    predict_sentiment()

    # Step 4: Correlate sentiment with stock performance
    correlate_sentiment_with_stock()
