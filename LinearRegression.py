import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = 0
        self.bias = 0

        for _ in range(self.n_iters):
            y_hat = x.T[0]*self.weights + self.bias #y = wx + b
            diff = y_hat - y
            dw = (1/n_samples) * np.sum(diff*x.T[0])
            db = (1/n_samples) * np.sum(diff)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        y_hat = x.T[0]*self.weights + self.bias
        return y_hat

if __name__ == "__main__":
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(n_iters= 100000)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)

    print(np.mean((y_test - predicted)**2))

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()