# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
# Fetch the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# Load the images and target labels
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

n_classes = target_names.shape[0]
print(f"Number of classes: {n_classes}")
print(f"Images shape: {lfw_people.images.shape}")
print(f"Number of samples: {X.shape[0]}")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Apply PCA (Principal Component Analysis) for dimensionality reduction
n_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# Train a Support Vector Machine (SVM) classifier
svc = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='auto')
svc.fit(X_train_pca, y_train)
# Make predictions on the test set
y_pred = svc.predict(X_test_pca)
# Display classification report and confusion matrix
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
# Plot some results
fig, ax = plt.subplots(2, 5, figsize=(15, 8))
for i in range(10):
    ax[i // 5, i % 5].imshow(X_test[i].reshape(50, 37), cmap='gray')
    ax[i // 5, i % 5].set_title(f'True: {target_names[y_test[i]]}\nPred: {target_names[y_pred[i]]}')
    ax[i // 5, i % 5].axis('off')
