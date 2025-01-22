import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def predict_sentiment(stock_symbol):
    model_path = "./models/sentiment_model"
    
    # Load test data
    test_data = pd.read_csv("data/test.csv")
    stock_tweets = test_data[test_data['Symbol'] == stock_symbol]
    
    if len(stock_tweets) == 0:
        print(f"No tweets found for {stock_symbol}")
        return
        
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    predictions = []
    for tweet in stock_tweets['Cleaned_Tweet']:
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=64)
        outputs = model(**inputs)
        prediction = torch.softmax(outputs.logits, dim=1)
        predictions.append(prediction[0][1].item())  # Probability of positive sentiment
    
    stock_tweets['Predicted_Sentiment'] = predictions
    stock_tweets.to_csv("data/predictions.csv", index=False)
    print("Sentiment prediction completed!")


if __name__ == "__main__":
    predict_sentiment()
