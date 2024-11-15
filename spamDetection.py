import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
sms = pd.read_csv('C:/Users/Admin/OneDrive/Desktop/AI IN CS/spam-naive bayes and NLP.csv', sep=',', 
names=["type", "text"], encoding='ISO-8859-1')
# Ensure all entries in 'type' and 'text' are filled and clean
sms['type'] = sms['type'].fillna('') # Fill any NaN values in type with empty strings
sms['text'] = sms['text'].fillna('').astype(str) # Fill NaN in text and ensure it is string
# Filter rows where 'type' is empty
sms = sms[sms['type'] != '']
# Preprocessing functions
def get_tokens(text):
    tokens = word_tokenize(text)
    return tokens
def get_lemmas(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas
# Apply tokenization and lemmatization
sms['tokens'] = sms['text'].apply(get_tokens)
sms['lemmas'] = sms['tokens'].apply(get_lemmas)
# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sms['text']) # Use original text data for vectorization
y = sms['type'] # Labels (spam or ham)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
# Predict on the test set
y_pred = classifier.predict(X_test)
# Evaluate the model with zero_division set to 1
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1) # Adds zero_division=1 to handle undefined 
metrics
# Output the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)