import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr #Learning rate
        self.n_iters = n_iters #No Interations
        self.weights = None #w
        self.bias = None #b
    
    def fit(self, x, y):
        #initilization
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_hat = np.dot(x, self.weights) + self.bias #y = wx + b
            diff = y_hat - y
            #update gradient
            dw = (1 / n_samples) * np.dot(x.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        y_hat = np.dot(x, self.weights) + self.bias
        return y_hat

if __name__ == "__main__":
    #get the sample data
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )
     #split into training data set and test data set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(n_iters= 100000)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)

    #mean squared error
    print(np.mean((y_test - predicted)**2))

    #drawin the line of best fit to the data
    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()