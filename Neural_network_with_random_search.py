__author__ = 'Stefan Chulski'

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random
import copy
from itertools import permutations
import pickle


def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


'''def splitDataset(dataset, splitRatio_test, splitRatio_validation):
    trainSize = int(len(dataset) * splitRatio_test)
    trainSet = []
    validationSize = int(len(dataset) * splitRatio_validation)
    validationSet = []
    copy = list(dataset)
    while len(validationSet) < validationSize:
        index = random.randrange(len(copy))
        validationSet.append(copy.pop(index))

    # while len(trainSet) < trainSize:
    #     trainSet.append(copy.pop(0))
    # while len(validationSet) < validationSize:
    #     validationSet.append(copy.pop(0))
    return [trainSet, validationSet, copy]'''


def determine_fold_sizes(size, k):
    fold_size = list()
    fold_size.append(int(size / k))
    for i in range(1, k):
        if i == k - 1:
            fold_size.append(size - (k - 1) * int(size / k))
        else:
            fold_size.append(int(size / k))
    return fold_size


def split_set_to_train_test(dataset, train_size):
    copy = list(dataset)
    train_set = []
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    test_set = copy
    return train_set, test_set


def split_train_to_k_folds(data, fold_size):
    index = 0
    folds = list()
    for size in fold_size:
        fold = data[index:index+size]
        folds.append(fold)
        index += size
    return folds


def kfold_split(dataset, train_ratio, k):
    train_size = int(len(dataset) * train_ratio)
    train_set, test_set = split_set_to_train_test(dataset, train_size)
    fold_size = determine_fold_sizes(train_size, k)
    folds = split_train_to_k_folds(train_set, fold_size)
    return folds, test_set





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


def ReLU(x):
    return x * (x > 0)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class Config:
    nn_input_dim = 8  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters
    epsilon = 0.0001  # learning rate for gradient descent
    reg_lambda = 0.0001  # regularization strength


def weight_square(W):
    sum = 0
    for val in W:
        sum += np.sum(np.square(val))
    return sum


# Helper function to evaluate the total loss on the dataset
def calculate_loss(W, b, X, y, nn_layers):
    num_examples = len(X)  # training set size
    probs, dummy = forward_propagation(W, b, X, nn_layers)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss
    data_loss += Config.reg_lambda / 2 * weight_square(W)
    return 1. / num_examples * data_loss


def predict(W, b, X, nn_layers):
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
            W.append(np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim))
            b.append(np.zeros((1, nn_hdim)))

    return W, b


def set_random_new(layers):
    W = []
    b = []
    np.random.seed(0)
    loop = len(layers) + 1

    for i in range(0, loop):
        if i == 0:
            W.append(np.random.randn(Config.nn_input_dim, layers[i]) / np.sqrt(Config.nn_input_dim))
            b.append(np.zeros((1, layers[i])))
        elif i == (loop - 1):
            W.append(np.random.randn(layers[i - 1], Config.nn_output_dim) / np.sqrt(layers[i - 1]))
            b.append(np.zeros((1, Config.nn_output_dim)))
        else:
            W.append(np.random.randn(layers[i - 1], layers[i]) / np.sqrt(layers[i - 1]))
            b.append(np.zeros((1, layers[i])))

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
            dW = a[(loop-2-i)].T.dot(delta)
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
            dW = a[(loop-2-i)].T.dot(delta)
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
# - nn_layers: Number of hidden layers
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, X1, y, y1, layers, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)
    nn_layers = len(layers) + 1

    w, b = set_random_new(layers)

    test = []
    validation = []
    best_loss = 999999999

    # Gradient descent. For each batch.
    for i in range(0, num_passes):
        print(i)
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
        # keep best weights
        train_loss = calculate_loss(w, b, X, y, nn_layers)
        valid_loss = calculate_loss(w, b, X1, y1, nn_layers)
        if valid_loss < best_loss or valid_loss < train_loss:
            w_best = copy.deepcopy(w)
            b_best = copy.deepcopy(b)
            best_loss = valid_loss
        # Optionally print the loss.
        if print_loss and i % 1000 == 0:
            print("Loss new after iteration %i: %f" % (i, calculate_loss(w, b, X, y, nn_layers)))
            test.append(calculate_loss(w, b, X, y, nn_layers))
            validation.append(calculate_loss(w, b, X1, y1, nn_layers))
            if best_loss > calculate_loss(w, b, X1, y1, nn_layers) or calculate_loss(w, b, X1, y1, nn_layers) < calculate_loss(w, b, X, y, nn_layers):
                w_best = copy.deepcopy(w)
                b_best = copy.deepcopy(b)
                best_loss = calculate_loss(w, b, X1, y1, nn_layers)
                print(i)

    return test, validation, w_best, b_best


def replace_with_median(df, col_name, nan_value):
    series = df[col_name]
    median = series[series!=nan_value].median()
    df[col_name].replace(nan_value, median, inplace=True)


def random_net_config(nodes_range, layers_range, learning_rate_range):
    layers = random.randint(layers_range[0], layers_range[1])
    nodes = list()
    for i in range(layers):
        nodes.append(random.randint(nodes_range[0], nodes_range[1]))
    learning_rate = random.uniform(learning_rate_range[0], learning_rate_range[1])
    return nodes, learning_rate


def randomized_search(iterations, data, ranges, k, train_ratio):
    weights = list()
    biases = list()
    plots = list()
    parameters = list()
    validations = list()
    nodes_range = ranges['nodes']
    layers_range = ranges['layers']
    learning_rate_range = ranges['learning_rate']
    folds, test = kfold_split(data, train_ratio, k)
    permut_it = permutations(range(k))
    permut = list(permut_it)
    X3, y3 = summarize(test)
    for i in range(iterations):
        layers, Config.epsilon = random_net_config(nodes_range, layers_range, learning_rate_range)
        epsilon = Config.epsilon
        W = list()
        B = list()
        valid = list()
        for p in permut:
            training = folds[p[0]] + folds[p[1]]
            validation = folds[p[2]]
            X1, y1 = summarize(training)
            X2, y2 = summarize(validation)
            train_plot, validation_plot, w, b = build_model(np.asarray(X1).T, np.asarray(X2).T,
                                                            np.asarray(y1).astype(int),
                                                            np.asarray(y2).astype(int), layers, print_loss=False)
            plots.append({'train_plot': train_plot, 'validation_plot': validation_plot})
            v = predict(w, b, np.asarray(X2).T, len(layers) + 1)
            valid_accuracy = accuracy_percent(v, y2)
            valid.append(valid_accuracy)
            W.append(w)
            B.append(b)
        validation_ave = mean(valid)
        parameters.append([layers, epsilon])
        validations.append(validation_ave)
        weights.append(W)
        biases.append(B)
    return X3, y3, plots, parameters, validations, weights, biases




'''def randomized_search(iterations, train, validation, test, ranges):
    models = list()
    predictions = list()
    plots = list()
    accuracies = list()
    X1 = train[0]
    y1 = train[1]
    X2 = validation[0]
    y2 = validation[1]
    X_test = test[0]
    y_test = test[1]
    nodes_range = ranges[0]
    layers_range = ranges[1]
    learning_rate_range = ranges[2]
    for i in range(iterations):
        layers, Config.epsilon = random_net_config(nodes_range, layers_range, learning_rate_range)
        train_plot, validation_plot, w, b = build_model(np.asarray(X1).T, np.asarray(X2).T, np.asarray(y1).astype(int),
                                                        np.asarray(y2).astype(int), layers, print_loss=True)
        prediction = predict(w, b, np.asarray(X_test).T, len(layers) + 1)
        models.append([w, b])
        plots.append({'train_plot':train_plot, 'validation_plot':validation_plot})
        predictions.append(prediction)
        accuracy = accuracy_percent(prediction, y_test)
        accuracies.append(accuracy)
    return models, predictions, accuracies, plots'''


def accuracy_percent(prediction, correct_labels):
    predicted = 0
    samples_count = len(prediction)
    for i in range(0, samples_count ):
        if correct_labels[i] == prediction[i]:
            predicted += 1
    return (float(predicted)/samples_count)*100.


def main():
    K = 3
    ratio = 0.9
    iterations = 50
    data = loadCsv('diabetes.csv')
    parameter_ranges = {'nodes':[1, 7], 'layers':[1, 4], 'learning_rate':[0.001, 0.3]}
    X_test, y_test, plots, parameters, validations, weights, biases = randomized_search(iterations, data, parameter_ranges, K, ratio)
    max_valid_accuracy = max(validations)
    print('The best accuracy in validation was:', max_valid_accuracy)
    print('The parameter models for this are:', parameters[validations.index(max_valid_accuracy)])
    pickle.dump(plots, open("plots", "wb"))
    pickle.dump(parameters, open("parameters.p", "wb"))
    pickle.dump(validations, open("validations.p", "wb"))
    pickle.dump(weights, open("weights.p", "wb"))
    pickle.dump(biases,  open("biases.p", "wb"))
    pickle.dump(X_test,  open("X_test.p", "wb"))
    pickle.dump(y_test, open("y_test.p", "wb"))
    '''for plot in plots:
        plt.plot(plot['train_plot'], color='g', label='Training')
        plt.plot(plot['validation_plot'], color='r', label='Validation')
        plt.title('Training')
        plt.ylabel('Loss')
        plt.xlabel('Passes in 10^3 ')
        plt.legend()
        plt.show()'''

if __name__ == "__main__":
    main()