__author__ = 'Stefan Chulski'

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random


def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio_test, splitRatio_validation):
    trainSize = int(len(dataset) * splitRatio_test)
    trainSet = []
    validationSize = int(len(dataset) * splitRatio_validation)
    validationSet = []
    copy = list(dataset)
    while len(validationSet) < validationSize:
        index = random.randrange(len(copy))
        validationSet.append(copy.pop(index))
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    # while len(trainSet) < trainSize:
    #     trainSet.append(copy.pop(0))
    # while len(validationSet) < validationSize:
    #     validationSet.append(copy.pop(0))
    return [trainSet, validationSet, copy]


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def max_num(numbers):
    return max(numbers)


def summarize(dataset):
    sum1 = []
    for ind, attribute in enumerate(zip(*dataset)):
        sum1.append([])
        for i in range(0, len(attribute)):
            sum1[ind].append((attribute[i] - mean(attribute)) / mean(attribute))
            # sum1[ind].append((attribute[i]) / max(attribute))
    del sum1[-1]
    return sum1, attribute


def add_binary_columns(feature,df):
    values = sorted(df[feature].unique())
    for value in values:
        df[feature+"_"+str(value)] = df[feature].apply(lambda x: 1 if x==value else 0,1)
    return df


def ReLU(x):
    return x * (x > 0)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class Config:
    nn_input_dim = 8  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.0001  # learning rate for gradient descent
    reg_lambda = 0.0001  # regularization strength


def visualize(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    num_examples = len(X)  # training set size
    # W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4'], model['W5'], model['b5']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = tanh(z1)
    # a1 = ReLU(z1)
    # a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = tanh(z2)
    z3 = a2.dot(W3) + b3
    a3 = tanh(z3)
    z4 = a3.dot(W4) + b4
    a4 = tanh(z4)
    z5 = a4.dot(W5) + b5
    exp_scores = np.exp(z5)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)) + np.sum(np.square(W4)) + np.sum(np.square(W5)))
    return 1. / num_examples * data_loss


def weight_square(W):
    sum = 0
    for val in W:
        sum += np.sum(np.square(val))
    return sum

def calculate_loss_new(W, b, X, y):
    num_examples = len(X)  # training set size
    probs, dummy = forward_propagation(W, b, X, 5)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * weight_square(W)
    return 1. / num_examples * data_loss


def predict(model, X):
    # W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4'], model['W5'], model['b5']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = tanh(z1)
    # a1 = ReLU(z1)
    # a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = tanh(z2)
    z3 = a2.dot(W3) + b3
    a3 = tanh(z3)
    z4 = a3.dot(W4) + b4
    a4 = tanh(z4)
    z5 = a4.dot(W5) + b5
    exp_scores = np.exp(z5)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def predict_new(W, b, X, nn_layers):
    probs, dummy = forward_propagation(W, b, X, nn_layers)
    return np.argmax(probs, axis=1)


def set_random(nn_hdim, loop):
    W = []
    b = []
    np.random.seed(0)

    for i in range(0, loop):
        if i == 0:
            W.append(np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim))
            b.append(np.zeros((1, nn_hdim)))
        elif i == (loop - 1):
            W.append(np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim))
            b.append(np.zeros((1, Config.nn_output_dim)))
        else:
            W.append(np.random.randn(nn_hdim, nn_hdim) / np.sqrt(Config.nn_input_dim))
            b.append(np.zeros((1, nn_hdim)))

    return W, b


def forward_propagation(W, b, X, loop):
    a_all = []
    for i in range(0, loop):
        if i == 0:
            z = X.dot(W[i]) + b[i]
            a = tanh(z)
            a_all.append(a)
        elif i == (loop - 1):
            z = a.dot(W[i]) + b[i]
            exp_scores = np.exp(z)
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True), a_all
        else:
            z = a.dot(W[i]) + b[i]
            a = tanh(z)
            a_all.append(a)


def backpropagation_propagation(X, a, W, delta, loop):
    dW_all = []
    db_all = []
    for i in range(0, loop):
        if i == 0:
            dW = (a[(loop-2-i)].T).dot(delta)
            dW_all.append(dW)
            db = np.sum(delta, axis=0, keepdims=True)
            db_all.append(db)
            delta = delta.dot(W[(loop-1-i)].T) * (1 - np.power(a[(loop-2-i)], 2))
        elif i == (loop - 1):
            dW = np.dot(X.T, delta)
            dW_all.append(dW)
            db = np.sum(delta, axis=0)
            db_all.append(db)
            return dW_all, db_all
        else:
            dW = (a[(loop-2-i)].T).dot(delta)
            dW_all.append(dW)
            db = np.sum(delta, axis=0, keepdims=True)
            db_all.append(db)
            delta = delta.dot(W[(loop-1-i)].T) * (1 - np.power(a[(loop-2-i)], 2))


def regularization_terms(W, dW, loop):
    for i in range(0, loop):
        dW[loop - 1 - i] += Config.reg_lambda * W[i]
    return dW


def gradient_descent_update(W, dW, b, db, loop):
    for i in range(0, loop):
        W[i] += -Config.epsilon * dW[loop - 1 - i]
        b[i] += -Config.epsilon * db[loop - 1 - i]
    return W, b


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, X1, y, y1, nn_hdim, nn_layers, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)

    w, b = set_random(nn_hdim, nn_layers)

    test = []
    validation = []

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        probs, a = forward_propagation(w, b, X, nn_layers)

        # Backpropagation
        delta = probs
        delta[range(num_examples), y] -= 1
        dW, db = backpropagation_propagation(X, a, w, delta, nn_layers)

        # Add regularization terms
        dW = regularization_terms(w, dW, nn_layers)

        # Gradient descent parameter update
        w, b = gradient_descent_update(w, dW, b, db, nn_layers)

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss new after iteration %i: %f" % (i, calculate_loss_new(w, b, X, y)))
            test.append(calculate_loss_new(w, b, X, y))
            validation.append(calculate_loss_new(w, b, X1, y1))

    return test, validation, w, b


def classify(X, y):
    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X, y)
    # return clf

    pass

def replace_with_median(df, col_name, nan_value):
    series = df[col_name]
    median = series[series!=nan_value].median()
    df[col_name].replace(nan_value, median, inplace=True)


def main():
    # print('Generated data:')
    # print(X)
    # print(np.asarray(X1))
    # print(np.asarray(y1))
    # print(y)
    # model = build_model(X, y, 3, print_loss=True)
    # print(predict(model, X))
    # visualize(X, y, model)
    data = loadCsv('diabetes.csv')
    training, validation, test = splitDataset(data, 0.60, 0.20)

    # df = pd.read_csv('diabetes.csv')
    # replace_with_median(df, 'Glucose', 0)
    # replace_with_median(df, 'BloodPressure', 0)
    # replace_with_median(df, 'BMI', 0)
    # replace_with_median(df, 'SkinThickness', 0)
    # replace_with_median(df, 'Insulin', 0)
    #
    # file = open('diabetes_new.csv', 'w')
    #
    # print(df)
    # df.to_csv(file, sep=' ', encoding='utf-8')
    # file.close()

    nn_layers = 5

    X1, y1 = summarize(training)
    X2, y2 = summarize(validation)

    train_plot, validation_plot, w, b = build_model(np.asarray(X1).T, np.asarray(X2).T, np.asarray(y1).astype(int), np.asarray(y2).astype(int), 5, nn_layers, print_loss=True)

    X1, y1 = summarize(test)

    print('Predict')
    final = predict_new(w, b, np.asarray(X1).T, nn_layers)

    print(final)
    print(np.asarray(y1).astype(int))

    predicted = 0
    for i in range(0, len(final)):
        if y1[i] == final[i]:
            predicted += 1

    print("Predicted: ", predicted*100/len(final), "%")

    # Plot final graph
    plt.plot(train_plot, color='g', label='train plot')
    plt.plot(validation_plot, color='r', label='validation plot')
    plt.title('Training')
    plt.ylabel('Loss')
    plt.xlabel('Passes in 10^3 ')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()