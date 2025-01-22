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
    data_path = 'data/stock_tweets.csv'
    
    try:
        data = pd.read_csv(data_path, names=['Tweet', 'Symbol', 'Company'])
        print(f"Successfully loaded data from {data_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        return

    data["Cleaned_Tweet"] = data["Tweet"].apply(clean_tweet)
    data["Sentiment"] = data["Cleaned_Tweet"].apply(lambda x: "positive" if "profit" in x or "buy" in x else "negative")
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    
    print("Data preprocessing completed!")

if __name__ == "__main__":
    preprocess_data()
