import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt

class DigitRecoginition:
    def __init__(self, lr = 0.001, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def init_param(self, features):
        W1 = np.random.rand(10, features) - 0.5
        b1 = np.random.rand(10,1) - 0.5
        W2 = np.random.rand(10,10) - 0.5
        b2 = np.random.rand(10,1) - 0.5
        return W1, b1, W2, b2
    
    def ReLU(self, z):
        return np.maximum(z,0)
    
    def derivative_ReLU(self, z):
         return z > 0
    
    def softMax(self, z):
        exp = np.exp(z - np.max(z))
        output = exp / exp.sum(axis=0)
        return output
    
    def output_fill(self, Y, m):
        output = np.zeros((10, m))
        output[Y,np.arange(Y.size)] = 1
        return output

    def forward_prop(self, X, W1, b1, W2, b2):
        z1 = np.matmul(W1, X) + b1
        a1 = self.ReLU(z1)
        z2 = np.matmul(W2, a1) + b2
        a2 = self.softMax(z2)
        return z1, a1, z2, a2
    
    def back_prop(self, X, Y, a1, a2, W2, z1, m):
        y_prob = self.output_fill(Y, m)
        dz2 = a2 - y_prob
        dw2 = (1/m) * np.matmul(dz2, a1.T)
        db2 = (1/m) * np.sum(dz2)
        dz1 = np.dot(W2.T, dz2)*self.derivative_ReLU(z1)
        dw1 = (1/m) * np.matmul(dz1, X.T)
        db1 = (1/m) * np.sum(dz1)
        return dw1, db1, dw2, db2

    def fit(self, X, Y):
        #initilization
        n_features, n_samples = X.shape
        W1, b1, W2, b2 = self.init_param(n_features)

        # gradient descent
        for i in range(self.n_iter):
            #Calculate first
            z1, a1, z2, a2 = self.forward_prop(X, W1, b1, W2, b2)
            dw1, db1, dw2, db2 = self.back_prop(X, Y, a1, a2, W2, z1, n_samples)

            #Update parameters
            W1 -= self.lr * dw1
            b1 -= self.lr * db1
            W2 -= self.lr * dw2
            b2 -= self.lr * db2

            if (i+1) % int(self.n_iter/10) == 0:
                print(f"Iteration: {i+1} / {self.n_iter}")
                prediction = np.argmax(a2, 0) # get the largest number on each column (aka only choose the prediction number of each training set)
                print(f'{self.get_accuracy(prediction, Y):.3%}')

        return W1, b1, W2, b2
    
    def get_accuracy(self, Y_predict, Y_actual):
        return np.sum(Y_predict == Y_actual) / Y_actual.size

    def predict(self, X, W1 ,b1, W2, b2):
        _, _, _, A2 = self.forward_prop(X, W1, b1, W2, b2)
        return np.argmax(A2, 0)

if __name__ == "__main__":
    SCALE_FACTOR = 255
    print("check start")
    data_train = pd.read_csv('train.csv')
    data_test = pd.read_csv('test.csv')
    print("import data done")
    #transform to numpy
    data_train = np.array(data_train).T
    X_test = np.array(data_test).T/SCALE_FACTOR # scale so that does not overflow

    X_train, Y_train = data_train[1:]/SCALE_FACTOR, data_train[0] # scale so that does not overflow
    print("scale data done")   
    model = DigitRecoginition(lr = 0.45, n_iter = 10000)
    W1, b1, W2, b2 = model.fit(X_train, Y_train)
    print("train data done")  


    Y_predicted = model.predict(X_test, W1, b1, W2, b2)

    with open('submission.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImageId","Label"])
        for i in range(Y_predicted.size):
            writer.writerow([i+1,Y_predicted[i]])

    print("Output complete")



