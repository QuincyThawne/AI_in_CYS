import numpy as np
class NaiveBayes:
    def fit(self, X, y):
        # Separate data by class and calculate mean, variance, and prior for each class
        self.classes = np.unique(y)
        self.stats = {
        cls: {
        "mean": X[y == cls].mean(axis=0),
        "var": X[y == cls].var(axis=0),
        "prior": len(X[y == cls]) / len(X)
        }
        for cls in self.classes
        }
    def _gaussian_density(self, mean, var, x):
        # Gaussian probability density function
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
    def predict(self, X):
        # Predict class for each sample in X
        y_pred = []
        for x in X:
            posteriors = []
            for cls, params in self.stats.items():
                prior = np.log(params["prior"])
                likelihood = np.sum(np.log(self._gaussian_density(params["mean"], params["var"], x)))
                posteriors.append(prior + likelihood)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)
# Example usage
X = np.array([[1, 2], [2, 3], [3, 4], [6, 5], [7, 8], [8, 9]])
y = np.array([0, 0, 0, 1, 1, 1]) # Labels: 0 or 1
model = NaiveBayes()
model.fit(X, y)
X_test = np.array([[2, 3], [7, 6]])
print("Predictions:", model.predict(X_test))