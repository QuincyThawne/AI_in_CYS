# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 1. Simple Supervised Classifier (Logistic Regression)
log_reg_clf = LogisticRegression(max_iter=1000, random_state=42)
log_reg_clf.fit(X_train, y_train)
log_reg_pred = log_reg_clf.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)

# 2. Bagging Classifier
bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging_clf.fit(X_train, y_train)
bagging_pred = bagging_clf.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_pred)
# 3. Boosting Classifier (AdaBoost)
boosting_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, 
random_state=42)
boosting_clf.fit(X_train, y_train)
boosting_pred = boosting_clf.predict(X_test)
boosting_accuracy = accuracy_score(y_test, boosting_pred)
# 4. Ensemble Learner (Voting Classifier)
# Using Logistic Regression, Decision Tree, and K-Nearest Neighbors as base models
knn_clf = KNeighborsClassifier()
voting_clf = VotingClassifier(
 estimators=[
 ('lr', log_reg_clf),
 ('dt', DecisionTreeClassifier()),
 ('knn', knn_clf)
 ],
 voting='hard' # 'hard' for majority voting; 'soft' would use probabilities
)
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_pred)
# Print the accuracies
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Bagging Classifier Accuracy:", bagging_accuracy)
print("Boosting Classifier Accuracy:", boosting_accuracy)
print("Voting Classifier (Ensemble) Accuracy:", voting_accuracy)
