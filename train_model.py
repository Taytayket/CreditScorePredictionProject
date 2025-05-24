from data_pipeline import ppl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_process import load_data, clean_and_engineer, split_features, scale_features

import json
import os

class LogisticRegression:
    """
    Custom implementation of logistic regression using NumPy.
    Includes sigmoid activation, batch gradient descent, and binary classification.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Sigmoid function: maps real values into [0,1] for probability interpretation.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the model using batch gradient descent.
        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features)
            y (ndarray): Target vector of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Gradient calculation
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        """
        Predict probability scores for inputs.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        """
        Predict class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using Accuracy, F1 score, and AUC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("Custom Logistic Regression Performance:")
    print(f"   - Accuracy: {acc:.4f}")
    print(f"   - F1 Score: {f1:.4f}")
    print(f"   - AUC Score: {auc:.4f}")

    return acc, f1, auc


if __name__ == "__main__":
    # Step 1: Load and pre-process data
    print("Loading and pre-processing data...")
    df = ppl("data/accepted_2007_to_2018Q4.csv")
    X_train, X_test, y_train, y_test = split_features(df)
    X_train, X_test, _ = scale_features(X_train, X_test)

    # Step 2: Train model
    print("Training...")
    model = LogisticRegression(learning_rate=0.05, n_iters=1500)
    model.fit(X_train, y_train)

    # Step 3: Evaluate performance
    evaluate_model(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)

    model_dict = {
        "weights": model.weights.tolist(),
        "bias": model.bias
    }

    # Step 4: Save model
    with open("models/logistic_model.json", "w") as f:
        json.dump(model_dict, f)

    print("Model saved to models/logistic_model.json")