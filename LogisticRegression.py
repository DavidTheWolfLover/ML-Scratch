import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LogisticRegression:
    def __init__(self, lr = 0.001, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        #initilization
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iter):
            f = np.dot(x, self.weights) + self.bias
            y_hat = 1 / (1 + np.exp(-f))

            #update paramaters
            dw = (1/n_samples) * np.dot(x.T, y_hat - y)
            db = (1/n_samples) * np.sum(y_hat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        f = np.dot(x, self.weights) + self.bias
        y_hat = 1 / (1 + np.exp(-f))
        y_hat = np.array([1 if i > 0.5 else 0 for i in y_hat])
        return y_hat

if __name__ == "__main__":
    #get the sample data
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

     #split into training data set and test data set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    #learning phase
    regressor = LogisticRegression(lr=0.0001, n_iter=10000)
    regressor.fit(X_train, y_train)

    #prediction phase
    predictions = regressor.predict(X_test)
    accuracy = np.sum(y_test == predictions) / len(y_test)   

    print("Prediction accuracy:", accuracy)