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
    N_train = 10
    N_test = 100
    N = N_train + N_test
    x_train_ = np.random.rand(N_train) * 0.6
    mu_train = np.sin(2 * np.pi * x_train_)
    g = 0.01
    t_train_ = np.random.normal(mu_train, g)
    x_test_ = np.arange(0.0, 1.0, 1 / N_test)
    mu_test_ = np.sin(2 * np.pi * x_test_)
    t_test_ = np.random.normal(mu_test_, g)

    r_train_ = np.identity(N_train) * 0.01
    r_test_ = np.identity(N_test) * 0.01

    kernel_ = GaussianKernel(param_=np.array([1., 1.]))
    regression = GaussianProcessRegression(kernel=kernel_, x_train=x_train_, t_train=t_train_, r_train=r_train_)
    regression.fit(learning_rate=0.1, max_iter=100)
    mu_test_, sigma_test_ = regression.predict_dist(x_test_, r_test_)

    sigma_test_ = np.sqrt(sigma_test_)

    fig1 = plt.figure()
    # plt.scatter(x_test_, t_test_, color='g')
    plt.scatter(x_train_, t_train_, color='r')
    plt.plot(x_test_, mu_test_, color='b')
    plt.fill_between(x_test_, mu_test_ - np.diag(sigma_test_), mu_test_ + np.diag(sigma_test_), color="c", alpha=0.5)
    plt.ylim(-2.5, 2.5)
    plt.show()
