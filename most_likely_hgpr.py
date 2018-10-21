import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from gpr import GaussianKernel
from gpr import GaussianProcessRegression


class MostLikelyHGPR(object):
    def __init__(self, kernel, x_train, x_test, t_train, r_train, r_test, var_r_train, var_r_test):
        self.kernel = kernel
        self.x_train = x_train
        self.x_test = x_test
        self.t_train = t_train
        self.r_train = r_train
        self.r_test = r_test
        self.var_r_train = var_r_train
        self.var_r_test = var_r_test
        self.regression = GaussianProcessRegression(kernel=self.kernel, x_train=self.x_train, t_train=self.t_train \
                                                         , r_train=self.r_train)
        self.var_regression = GaussianProcessRegression(kernel=self.kernel, x_train=self.x_train \
                                                        , t_train=np.diag(self.r_train), r_train=self.var_r_train)

    def fit(self, learning_rate_gpr=0.1, max_iter_sampling=100, max_iter_gpr=100, sampling_num=100):
        self.regression.fit(learning_rate=learning_rate_gpr, max_iter=max_iter_gpr)
        mu_train, sigma_train = self.regression.predict_dist(x_test=self.x_train, r_test=self.r_train)
        try:
            for i in range(max_iter_sampling):
                r_before = np.array(self.r_train)
                for j in range(self.x_train.size):
                    if i == 0:
                        t_train_predict = np.random.normal(mu_train[j], np.sqrt(np.diag(sigma_train))[j], sampling_num)
                    else:
                        t_train_predict = np.random.normal(mu_train[j], np.sqrt(np.diag(self.r_train))[j], sampling_num)
                    r_predict = 0.5 * (1.0 / float(sampling_num)) \
                                * np.dot(self.t_train[j] - t_train_predict, self.t_train[j] - t_train_predict)
                    self.r_train[j, j] = r_predict
                if np.dot(np.diag(self.r_train) - np.diag(r_before), np.diag(self.r_train) - np.diag(r_before)) \
                        / np.diag(self.r_train).size < 1e-2:
                    self.var_regression.fit(learning_rate=learning_rate_gpr, max_iter=max_iter_gpr)
                    break
        except:
            raise

    def predict(self):
        var_mu_test, var_sigma_test = self.var_regression.predict_dist(self.x_test, self.var_r_test)
        for i in range(var_mu_test.size):
            self.r_test[i, i] = var_mu_test[i]
        self.regression.fit(learning_rate=100, max_iter=100)
        mu_test, sigma_test = self.regression.predict_dist(self.x_test, self.r_test)
        return mu_test, sigma_test


if __name__ == '__main__':
    print(__file__)
