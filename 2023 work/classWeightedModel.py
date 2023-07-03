import numpy as np
from collections import Counter

class WeightedKNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            # Calculate the distance between the test point and all training points
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            
            # Sort the distances and get the indices of the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            
            # Get the classes of the k nearest neighbors
            nearest_classes = self.y[nearest_indices]
            
            # Calculate the inverse class size for each class
            class_sizes = Counter(nearest_classes)
            inverse_sizes = {c: 1.0 / class_sizes[c] for c in class_sizes}
            
            # Weight the neighbor counts by inverse class size
            weighted_counts = Counter(nearest_classes)
            for c in weighted_counts:
                weighted_counts[c] *= inverse_sizes[c]
                
            # Convert neighbor counts into the fraction of each class
            total_count = sum(weighted_counts.values())
            class_fractions = {c: weighted_counts[c] / total_count for c in weighted_counts}
            
            # Predict the class with the highest fraction
            y_pred[i] = max(class_fractions, key=class_fractions.get)
        return y_pred


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Initialize the weighted k-NN classifier with k=5
clf = WeightedKNN(k=5)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the classes of the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
