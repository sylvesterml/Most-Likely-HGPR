import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from gpr import GaussianKernel
from gpr import GaussianProcessRegression

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
    plt.savefig('test_gpr.png')
