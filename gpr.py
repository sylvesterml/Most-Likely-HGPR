import numpy as np
import matplotlib.pylab as plt


class GaussianKernel(object):

    def __init__(self, param_):
        assert np.shape(param_) == (2,)
        self.param_ = param_

    def get_param_(self):
        return np.copy(self.param_)

    def __call__(self, x, y):
        return self.param_[0] * np.exp(-0.5 * self.param_[1] * (x - y) ** 2)

    def delta_param_(self, x, y):
        sq_diff = (x - y) ** 2
        delta_0 = np.exp(-0.5 * self.param_[1] * sq_diff)
        delta_1 = -0.5 * sq_diff * delta_0 * self.param_[0]
        return delta_0, delta_1

    def update_param_(self, updates):
        assert np.shape(updates) == (2,)
        self.param_ += updates


class GaussianProcessRegression(object):

    def __init__(self, kernel, x_train, t_train, r_train):
        self.kernel = kernel
        self.x_train = x_train
        self.t_train = t_train
        self.r_train = r_train

    def fit(self, learning_rate=0.1, max_iter=100):
        for i in range(max_iter):
            if i % 10:
                learning_rate /= 2.0
            param_ = self.kernel.get_param_()
            k_train = self.kernel(*np.meshgrid(self.x_train, self.x_train))
            c_train = k_train + self.r_train
            self.c_train_inv = np.linalg.pinv(c_train)
            delta_c = self.kernel.delta_param_(*np.meshgrid(self.x_train, self.x_train))
            updates = np.array([(-0.5) * np.trace(np.dot(self.c_train_inv, delta_c_i)) \
                                + 0.5 * np.dot(np.dot(np.dot(np.dot(self.t_train.T, self.c_train_inv), delta_c_i), \
                                                      self.c_train_inv), self.t_train) for delta_c_i in delta_c])
            self.kernel.update_param_(learning_rate * updates)
            if np.allclose(param_, self.kernel.get_param_()):
                break
        else:
            print('GPR\'s parameters may not have converged')

    def predict_dist(self, x_test, r_test):
        k_train_test = self.kernel(*np.meshgrid(x_test, self.x_train, indexing='ij'))
        k_test = self.kernel(*np.meshgrid(x_test, x_test))
        c_test = k_test + r_test
        mu_test = np.dot(np.dot(k_train_test, self.c_train_inv), self.t_train)
        sigma_test = c_test - np.dot(np.dot(k_train_test, self.c_train_inv), k_train_test.T)
        return mu_test, sigma_test


if __name__ == '__main__':
    print(__file__)
