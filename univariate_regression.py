
import os
import numpy as np
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from bgd import MyBGDRegression
from utils import read_data


def load_data(filename, input_var_name, output_var_name):
    data, data_names = read_data(filename)
    selected_var = data_names.index(input_var_name)
    inputs = [float(data[i][selected_var]) for i in range(len(data))]
    selected_output = data_names.index(output_var_name)
    outputs = [float(data[i][selected_output]) for i in range(len(data))]

    return inputs, outputs


def plot_data(x1, y1, x2=None, y2=None, x3=None, y3=None, title=None):
    plt.plot(x1, y1, 'ro', label='train data')
    if x2:
        plt.plot(x2, y2, 'b-', label='learnt model')
    if x3:
        plt.plot(x3, y3, 'g^', label='test data')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_histogram(x, var_name):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + var_name)
    plt.show()


def univariate_regression():
    curr_dir = os.getcwd()
    file_path = os.path.join(curr_dir, 'data', 'world-happiness-report-2017.csv')

    inputs, outputs = load_data(file_path, 'Economy..GDP.per.Capita.', 'Happiness.Score')

    plot_histogram(inputs, 'capita GDP')
    plot_histogram(outputs, 'Happiness score')

    # linearity check
    plot_data(inputs, outputs, [], [], [], [], 'capita vs. hapiness')

    # split data into training data (80%) and testing data (20%)
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]

    train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    test_sample = [i for i in indexes if not i in train_sample]

    train_inputs = [inputs[i] for i in train_sample]
    train_outputs = [outputs[i] for i in train_sample]
    test_inputs = [inputs[i] for i in test_sample]
    test_outputs = [outputs[i] for i in test_sample]

    # plot_data(train_inputs, train_outputs, [], [], test_inputs, test_outputs, "train and test data")

    # train_inputs = train_inputs[:5]
    # train_outputs = train_outputs[:5]

    # training
    xx = [[el] for el in train_inputs]
    # regressor = linear_model.SGDRegressor(max_iter =  10000)
    regressor = MyBGDRegression()
    regressor.fit(xx, train_outputs)
    w0, w1 = regressor.intercept_, regressor.coef_[0]

    # plot the model
    points_num = 1000
    xref = []
    val = min(train_inputs)
    step = (max(train_inputs) - min(train_inputs)) / points_num
    for i in range(1, points_num):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]
    # plotData(train_inputs, train_outputs, xref, yref, [], [], title="train data and model")

    # makes predictions for test data
    # computedTestOutputs = [w0 + w1 * el for el in test_inputs]

    # makes predictions for test data (by tool)
    computedTestOutputs = regressor.predict([[x] for x in test_inputs])

    plot_data([], [], test_inputs, computedTestOutputs, test_inputs, test_outputs, "predictions vs real test data")

    # compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, test_outputs):
        error += (t1 - t2) ** 2
    error = error / len(test_outputs)
    print("prediction error (manual): ", error)

    error = mean_squared_error(test_outputs, computedTestOutputs)
    print("prediction error (tool): ", error)
