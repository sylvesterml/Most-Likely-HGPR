import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

N_train = 60
N_test = 100
N = N_train+N_test
x_train = np.random.rand(N_train)
# mu_ = np.sin(2 * np.pi * x)
# g = 0.16
# t = np.random.normal(mu_, g)
mu_ = 2.0 * (np.exp(-30.0 * (x_train - 0.25) ** 2.0) + np.sin(np.pi * x_train) ** 2) - 2.0
g = np.sin(2.0 * np.pi * x_train)
t_train = np.random.normal(mu_, np.exp(g))
x_test = np.arange(0.0, 1.0, 1/N_test)

# x_train, x_test, t_train, t_test = train_test_split(x, t, train_size=0.8, random_state=1)
# x_test = np.sort(x_test)

# x_train = np.random.rand(10) * 0.5 + 0.2
# mu_train = np.sin(2 * np.pi * x_train)
# g_train = 0.04
# t_train = np.random.normal(mu_train, g_train)

# x_test = np.arange(0, 1.0, 0.01)
# mu_test = np.sin(2 * np.pi * x_test)
# g_test = 0.04
# t_test = np.random.normal(mu_test, g_test)


C = np.zeros((N, N))
C_train = np.zeros((N_train, N_train))
C_train_inv = np.zeros((N_train, N_train))
K = np.zeros((N_train, N_train))
K_1 = np.zeros((N_test, N_train))
K_2 = np.zeros((N_test, N_test))
R = np.identity(N_train) * 0.16
R_1 = np.identity(N_test) * 0.16
param_ = np.ones(2)
before_param_ = np.zeros(2)
alpha_ = 1.0

def delta_C(x, param_, n):
    delta_C_0 = np.zeros((n, n))
    delta_C_1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            delta_C_0[i, j] = np.exp((-1.0) * (x[i] - x[j])**2 * param_[1] / 2.0)
            delta_C_1[i, j] = param_[0] * np.exp((-1.0) * (x[i] - x[j])**2 * param_[1] / 2.0) * (-1.0) * \
                              (x[i] - x[j])**2 / 2.0
    return delta_C_0, delta_C_1

cnt = 0
while(1):
    print(cnt)
    cnt += 1
    if (cnt % 10):
        alpha_ /= 2.0
    for i in range(N_train):
        for j in range(N_train):
            K[i, j] = param_[0] * np.exp((-1.0) * ((x_train[i] - x_train[j])**2) * param_[1] / 2.0)
            C[i, j] = K[i, j] + R[i, j]
            C_train[i, j] = K[i, j] + R[i, j]

    for i in range(N_test):
        for j in range(N_train):
            K_1[i, j] = param_[0] * np.exp((-1.0) * ((x_test[i] - x_train[j])**2) * param_[1] / 2.0)
            C[N_train+i, j] = K_1[i, j]
            C[j, N_train+i] = K_1[i, j]

    for i in range(N_test):
        for j in range(N_test):
            K_2[i, j] = param_[0] * np.exp((-1.0) * ((x_test[i] - x_test[j])**2) * param_[1] / 2.0)
            C[N_train+i, N_train+j] = K_2[i, j] + R_1[i, j]

    C_train_inv = np.linalg.pinv(C_train)
    delta_C_train_0, delta_C_train_1 = delta_C(x_train, param_, N_train)
    delta_l_0 = (-0.5) * np.trace(np.dot(C_train_inv, delta_C_train_0)) + (0.5) * np.dot(np.dot(np.dot(np.dot(t_train.T, \
                                                                    C_train_inv), delta_C_train_0), C_train_inv), t_train)
    delta_l_1 = (-0.5) * np.trace(np.dot(C_train_inv, delta_C_train_1)) + (0.5) * np.dot(np.dot(np.dot(np.dot(t_train.T, \
                                                                    C_train_inv), delta_C_train_1), C_train_inv), t_train)
    before_param_[0] = param_[0]
    before_param_[1] = param_[1]
    param_[0] += alpha_ * delta_l_0
    param_[1] += alpha_ * delta_l_1

    if (np.absolute(param_[0] - before_param_[0]) < 0.0001):
        if (np.absolute(param_[1] - before_param_[1]) < 0.0001):
            break

print(param_)

m_test = np.dot(np.dot(K_1, C_train_inv), t_train)
sigma_test = K_2 + R_1 - np.dot(np.dot(K_1, C_train_inv), K_1.T)

fig1 = plt.figure()
# plt.scatter(x_test, t_test, color='g')
plt.scatter(x_train, t_train, color='r')
plt.plot(x_test, m_test, color='b')
plt.fill_between(x_test, m_test - np.diag(sigma_test), m_test + np.diag(sigma_test), color="c", alpha=0.5)
plt.ylim(-2.5, 2.5)
plt.show()
