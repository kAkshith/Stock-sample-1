import pandas as pd

def correlate_sentiment_with_stock(stock_symbol):
    data = pd.read_csv("data/predictions.csv")
    sentiment_score = data['Predicted_Sentiment'].mean()
    tweet_count = len(data)
    
    decision = "HOLD"
    if sentiment_score > 0.6:
        decision = "BUY"
    elif sentiment_score < 0.4:
        decision = "SELL"
    
    print(f"\nAnalysis for {stock_symbol}:")
    print(f"Number of tweets analyzed: {tweet_count}")
    print(f"Sentiment Score: {sentiment_score:.2f} -> Decision: {decision}")

if __name__ == "__main__":
    correlate_sentiment_with_stock()
