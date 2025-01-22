import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def predict_sentiment():
    # Load model and tokenizer
    model_path = "StockPredictionSentimentAnalysis/models/sentiment_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Load test data
    test_data = pd.read_csv("StockPredictionSentimentAnalysis/data/test.csv")

    predictions = []
    for tweet in test_data["Cleaned_Tweet"]:
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        predictions.append(pred)

    test_data["Predicted_Sentiment"] = predictions
    test_data.to_csv("StockPredictionSentimentAnalysis/data/predictions.csv", index=False)
    print("Sentiment prediction completed!")

if __name__ == "__main__":
    predict_sentiment()
