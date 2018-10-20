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
    N_train = 1200
    N_test = 200
    split_rate = 0.1
    N_train_1 = int(N_train * (1.0 - split_rate))
    N = N_train + N_test

    mu_test_ave = np.zeros(N_test)
    sigma_test_ave = np.zeros(N_test)
    count = 1

    x_train_ = np.arange(0.0, 1.0, 1 / N_train)
    x_test_ = np.arange(0.0, 1.0, 1 / N_test)
    mu_train = 2.0 * (np.exp(-30.0 * (x_train_ - 0.25) ** 2.0) + np.sin(np.pi * x_train_) ** 2) - 2.0
    g_train = np.sin(2.0 * np.pi * x_train_)
    t_train_ = np.random.normal(mu_train, np.exp(g_train))
    x_test_ = np.arange(0.0, 1.0, 1 / N_test)

    r_train_ = np.identity(N_train) * 0.01
    r_test_ = np.identity(N_test) * 0.01
    var_r_train_ = np.identity(N_train) * 0.01
    var_r_test_ = np.identity(N_test) * 0.01

    cnt = 0
    broke_num = 0
    while cnt < count:
        print(cnt)

        # x_train_1, x_train_2, t_train_1, t_train_2 = train_test_split(x_train_, t_train_, test_size=split_rate)

        try:
            kernel_ = GaussianKernel(param_=np.array([1., 1.]))
            regression = MostLikelyHGPR(kernel=kernel_, x_train=x_train_, x_test=x_test_, t_train=t_train_ \
                                        , r_train=r_train_, r_test=r_test_, var_r_train=var_r_train_, var_r_test=var_r_test_)
            regression.fit(learning_rate_gpr=0.1, max_iter_sampling=500, max_iter_gpr=1000, sampling_num=5000)
            cnt += 1
        except:
            print('Error')
            broke_num += 1
            if broke_num < 5 * count:
                continue
            else:
                break

        mu_test_, sigma_test_ = regression.predict()
        sigma_test_ = np.sqrt(sigma_test_)
        sigma_test_diag = np.diag(sigma_test_)
        sigma_test_diag.flags.writeable = True
        sigma_test_diag[np.isnan(sigma_test_diag)] = 0
        mu_test_ave += mu_test_
        sigma_test_ave += sigma_test_diag

    mu_test_ave /= cnt
    sigma_test_ave /= cnt

    # x_test_ = np.arange(0.0, 1.0, 1 / N_test)
    mu_test = 2.0 * (np.exp(-30.0 * (x_test_ - 0.25) ** 2.0) + np.sin(np.pi * x_test_) ** 2) - 2.0
    g_test = np.sin(2.0 * np.pi * x_test_)
    t_test_ = np.random.normal(mu_test, np.exp(g_test))
    g_test = np.sqrt(np.exp(g_test))

    print(sigma_test_ave)

    fig1 = plt.figure()
    plt.fill_between(x_test_, mu_test_ave - sigma_test_ave*2, mu_test_ave - sigma_test_ave*3 \
                     , color="c", alpha=0.2)
    plt.fill_between(x_test_, mu_test_ave + sigma_test_ave * 2, mu_test_ave + sigma_test_ave * 3 \
                 , color="c", alpha=0.2)
    plt.fill_between(x_test_, mu_test_ave - sigma_test_ave, mu_test_ave - sigma_test_ave * 2 \
                     , color="c", alpha=0.4)
    plt.fill_between(x_test_, mu_test_ave + sigma_test_ave, mu_test_ave + sigma_test_ave * 2 \
                     , color="c", alpha=0.4)
    plt.fill_between(x_test_, mu_test_ave - sigma_test_ave, mu_test_ave + sigma_test_ave, color='c', alpha=0.6)
    plt.scatter(x_test_, t_test_, color='g', s=10)
    # plt.scatter(x_train_, t_train_, color='r')
    plt.plot(x_test_, mu_test, color='r')
    plt.plot(x_test_, mu_test + g_test, color='hotpink')
    plt.plot(x_test_, mu_test - g_test, color='hotpink')
    plt.plot(x_test_, mu_test + g_test * 2, color='violet')
    plt.plot(x_test_, mu_test - g_test * 2, color='violet')
    plt.plot(x_test_, mu_test + g_test * 3, color='lightpink')
    plt.plot(x_test_, mu_test - g_test * 3, color='lightpink')
    plt.plot(x_test_, mu_test_ave, color='b')
    plt.ylim(-7.0, 7.0)
    plt.savefig('Yuan_Wahba_data30.png')
    plt.show()
