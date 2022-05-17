import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

from utils import read_data


def loadDataMoreInputs(filename, input_var_names, output_var_name):
    data, data_names = read_data(filename)
    selectedVariable1 = data_names.index(input_var_names[0])
    selectedVariable2 = data_names.index(input_var_names[1])
    inputs = [[float(data[i][selectedVariable1]), float(data[i][selectedVariable2])] for i in range(len(data))]
    selectedOutput = data_names.index(output_var_name)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData


def bivariate_regression():

    import os

    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'world-happiness-report-2017.csv')

    inputs, outputs = loadDataMoreInputs(filePath, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')

    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    trainOutputs, testOutputs = normalisation(trainOutputs, testOutputs)

    # identify (by training) the regressor

    # # use sklearn regressor
    # from sklearn import linear_model
    # regressor = linear_model.SGDRegressor()

    # using developed code
    from bgd import MyBGDRegression

    # model initialisation
    regressor = MyBGDRegression()

    regressor.fit(trainInputs, trainOutputs)
    # parameters of the liniar regressor
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print("learnt model: f(x) = ", w0, " + ", w1, " * x1 + ", w2, " * x2")

    # makes predictions for test data
    # computedTestOutputs = [w0 + w1 * el[0] + w2 * el[1] for el in testInputs]
    # makes predictions for test data (by tool)
    computedTestOutputs = regressor.predict(testInputs)

    # compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print('prediction error (manual): ', error)

    # TOOL SOLUTION
    print()
    regressor = SGDRegressor(max_iter=1000, tol=0.001)
    regressor.fit(trainInputs, trainOutputs)

    # parameters of the liniar regressor
    w0, w1, w2 = regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1]
    print("learnt model: f(x) = ", w0, " + ", w1, " * x1 + ", w2, " * x2")

    computedTestOutputs = regressor.predict(testInputs)

    from sklearn.metrics import mean_squared_error

    error = mean_squared_error(testOutputs, computedTestOutputs)
    print('prediction error (tool):   ', error)
