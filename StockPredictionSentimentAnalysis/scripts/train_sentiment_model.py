import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label,
        }

def train_model():
    # Load preprocessed data
    train_data = pd.read_csv("StockPredictionSentimentAnalysis/data/train.csv")

    # Encode labels
    label_mapping = {"positive": 1, "negative": 0}
    train_data["Sentiment"] = train_data["Sentiment"].map(label_mapping)

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Prepare dataset
    train_dataset = TweetDataset(
        train_data["Cleaned_Tweet"].tolist(),
        train_data["Sentiment"].tolist(),
        tokenizer,
        max_len=128,
    )

    # Load pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="StockPredictionSentimentAnalysis/models/sentiment_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained("StockPredictionSentimentAnalysis/models/sentiment_model")
    tokenizer.save_pretrained("StockPredictionSentimentAnalysis/models/sentiment_model")

    print("Model training completed!")

if __name__ == "__main__":
    train_model()
