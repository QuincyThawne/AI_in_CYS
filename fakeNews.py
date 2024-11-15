# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Step 1: Load the dataset
data = pd.read_csv('C:/Users/Admin/OneDrive/Desktop/AI IN CS/WELFake_Dataset.csv')
# Step 2: Preprocess the data
# Drop any null rows
data = data.dropna()
# Converting text to lowercase for consistency
data['text'] = data['text'].str.lower()
# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
# Step 4: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Step 5: Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
# Step 6: Make predictions on the test set
y_pred = model.predict(X_test_tfidf)
# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)