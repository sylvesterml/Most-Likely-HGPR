import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from gpr import GaussianKernel
from gpr import GaussianProcessRegression

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
