import pandas as pd

def correlate_sentiment_with_stock():
    # Load predictions
    data = pd.read_csv("StockPredictionSentimentAnalysis/data/predictions.csv")

    # Aggregate sentiment
    sentiment_score = data["Predicted_Sentiment"].mean()

    # Decision Logic
    if sentiment_score > 0.6:
        decision = "BUY"
    elif sentiment_score < 0.4:
        decision = "SELL"
    else:
        decision = "HOLD"

    print(f"Sentiment Score: {sentiment_score:.2f} -> Decision: {decision}")

if __name__ == "__main__":
    correlate_sentiment_with_stock()
