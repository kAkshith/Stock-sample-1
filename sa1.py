import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from datetime import datetime
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

def load_data(stock_symbol):
    data = pd.read_csv("stock_tweets.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    return data[data['Stock Name'] == stock_symbol]

def analyze_sentiment(tweets):
    sentiment_analyzer = pipeline("sentiment-analysis")
    results = []
    
    for tweet in tweets:
        try:
            result = sentiment_analyzer(tweet[:512])[0]  # BERT typically has 512 token limit
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            results.append(score)
        except Exception as e:
            results.append(0)  # Neutral sentiment for errors
            
    return results

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def visualize_results(data, sentiments):
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Add sentiments to dataframe
    data['Sentiment'] = sentiments
    
    # Convert continuous sentiments to binary for metrics
    y_true = [1 if score > 0.5 else 0 for score in data['Sentiment']]
    y_pred = [1 if score > 0 else 0 for score in data['Sentiment']]
    
    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
    
    # Plot 1: Sentiment distribution
    sns.histplot(data=data, x='Sentiment', bins=30, ax=ax1)
    ax1.set_title('Sentiment Distribution')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Count')
    
    # Plot 2: Sentiment over time
    data.set_index('Date')['Sentiment'].rolling('1D').mean().plot(ax=ax2)
    ax2.set_title('Sentiment Trend Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Average Sentiment')
    
    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # Plot 4: Metrics
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax4)
    ax4.set_title('Performance Metrics')
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSentiment Analysis Summary:")
    print(f"Average Sentiment: {data['Sentiment'].mean():.3f}")
    print(f"Positive Tweets: {(data['Sentiment'] > 0).sum()} ({(data['Sentiment'] > 0).mean()*100:.1f}%)")
    print(f"Negative Tweets: {(data['Sentiment'] < 0).sum()} ({(data['Sentiment'] < 0).mean()*100:.1f}%)")
    print(f"Total Tweets Analyzed: {len(data)}")
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

def main(stock_symbol):
    # Load and preprocess data
    data = load_data(stock_symbol)
    if data.empty:
        print(f"No tweets found for {stock_symbol}")
        return
        
    print(f"Analyzing {len(data)} tweets for {stock_symbol}...")
    
    # Analyze sentiment
    sentiments = analyze_sentiment(data['Tweet'].tolist())
    
    # Visualize results
    visualize_results(data, sentiments)

if __name__ == "__main__":
    stock_symbol = input("Enter stock symbol (e.g., TSLA): ").upper()
    main(stock_symbol)