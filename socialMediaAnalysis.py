from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import emoji
# Setup
nltk.download('vader_lexicon')
api_key = "AIzaSyCqeXS04S2KFXKZM9g0GFMiIp8YB4nET5E" # Replace with your YouTube API key
youtube = build('youtube', 'v3', developerKey=api_key)
sid = SentimentIntensityAnalyzer()
# Fetch comments from a YouTube video
def get_comments(video_id):
    comments, response = [], youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText", 
    maxResults=100).execute()
    while response:
        comments.extend([item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']])
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(part="snippet", videoId=video_id, 
            pageToken=response['nextPageToken'], textFormat="plainText", maxResults=100).execute()
        else:
            break
    return comments
    # Preprocess text and analyze sentiment
def analyze_sentiments(comments):
    results = []
    for comment in comments:
        processed = emoji.demojize(comment)
        score = sid.polarity_scores(processed)['compound']
        sentiment = 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
        results.append({'Comment': comment, 'Sentiment': sentiment, 'Score': score})
    return pd.DataFrame(results)
# Example Usage
video_id = "FN5_IVIaBx8" # Replace with actual video IDs
comments = get_comments(video_id)
sentiment_results = analyze_sentiments(comments)
# Display some comments and their sentiments
print("Sample Comments and Sentiments:")
print(sentiment_results.sample(5)) # Display a random sample of 5 comments
# Save results to CSV
sentiment_results.to_csv("all_comments_sentiment.csv", index=False)
# Performance Evaluation
from sklearn.metrics import accuracy_score, classification_report
# Example labeled test data for evaluation
test_data = [
 {"Comment": "Loved it!", "True_Sentiment": "Positive"},
 {"Comment": "Horrible!", "True_Sentiment": "Negative"},
 {"Comment": "It was okay.", "True_Sentiment": "Neutral"},
]
# Extract true sentiments and analyze predictions
true_labels = [item["True_Sentiment"] for item in test_data]
pred_labels = analyze_sentiments([item["Comment"] for item in test_data])['Sentiment']
# Evaluation
print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.2%}")
print("\nClassification Report:\n", classification_report(true_labels, pred_labels))