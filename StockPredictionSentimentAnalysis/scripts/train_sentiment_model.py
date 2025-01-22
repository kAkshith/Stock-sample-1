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

def create_dataset(data, tokenizer, max_len=64):  # Reduced from 128 to 64
    # Take a subset of data for faster training
    data = data.sample(n=min(1000, len(data)), random_state=42)
    texts = data['Cleaned_Tweet'].tolist()
    labels = [1 if label == 'positive' else 0 for label in data['Sentiment'].tolist()]
    return TweetDataset(texts, labels, tokenizer, max_len)

def train_model():
    # Load training data
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    
    # Use smaller BERT model
    model_name = 'prajjwal1/bert-tiny'  # Much smaller than bert-base-uncased
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    train_dataset = create_dataset(train_data, tokenizer)
    eval_dataset = create_dataset(test_data, tokenizer)
    
    # Define model save path
    model_save_path = "./models/sentiment_model"
    
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print("Model training completed!")

if __name__ == "__main__":
    train_model()
