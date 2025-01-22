import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_tweet(tweet):
    """Clean the tweet text by removing URLs, mentions, hashtags, and special characters."""
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = tweet.lower()
    return tweet.strip()

def preprocess_data():
    """Load and preprocess the data."""
    # Load the dataset
    data_path = "StockPredictionSentimentAnalysis/data/stock_tweets.csv"
    data = pd.read_csv(data_path)

    # Clean the tweets
    data["Cleaned_Tweet"] = data["Tweet"].apply(clean_tweet)

    # Label data (Example: Map positive/negative labels manually if no labels exist)
    # You can use a pretrained sentiment analyzer here or manually label a subset of data.
    data["Sentiment"] = data["Cleaned_Tweet"].apply(lambda x: "positive" if "profit" in x or "buy" in x else "negative")

    # Split data into training and testing sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Save processed data
    train.to_csv("StockPredictionSentimentAnalysis/data/train.csv", index=False)
    test.to_csv("StockPredictionSentimentAnalysis/data/test.csv", index=False)

    print("Data preprocessing completed!")

if __name__ == "__main__":
    preprocess_data()
