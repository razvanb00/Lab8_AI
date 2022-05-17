import matplotlib.pyplot as plt

class MyBGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x, y, l_rate=0.001, epoch_num=1500):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]  # beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]    #beta or w coefficients
        errors = []
        for epoch in range(epoch_num):
            # TBA: shuffle the training examples in order to prevent cycles
            error_sum = 0
            new_error = [0 for x in range(len(x[0]))]

            for i in range(len(x)):  # for each sample from the training data
                y_computed = self.eval(x[i])  # estimate the output
                crt_error = y_computed - y[i]  # compute the error for the current sample
                for j in range(len(x[0])):
                    new_error[j] += crt_error * x[i][j]
                error_sum += crt_error

            mean_error = error_sum / len(x)
            errors.append(abs(mean_error)) # collect the mean error from each epoch
            for k in range(len(new_error)):
                new_error[k] = new_error[k] / len(x)

            for j in range(0, len(x[0])):  # update the coefficients
                self.coef_[j] = self.coef_[j] - l_rate * new_error[j]
            self.coef_[len(x[0])] = self.coef_[len(x[0])] - l_rate * mean_error * 1

        # plot the errors
        plt.plot([x1 for x1 in range(epoch_num)], errors)
        plt.show()

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, x):
        y_comp = [self.eval(xi) for xi in x]
        return y_comp
