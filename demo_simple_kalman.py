

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

np.random.seed(100)

mu_0 = np.array([4, 4])
cov_0 = 0.05*np.eye(2)
N = 400
X = np.linspace(0, 5, N)
Y = np.linspace(0, 7, N)
X, Y = np.meshgrid(X, Y)
pos = np.dstack((X, Y))

# previous time step
rv = multivariate_normal(mu_0, cov_0)
Z = rv.pdf(pos)

# error distributions
cov_q = 0.1*np.eye(2)
q = np.random.multivariate_normal(np.array([0, 0]), cov_q)
cov_r = 0.1*np.eye(2)
r = np.random.multivariate_normal(np.array([0, 0]), cov_q)
# print(q)
# print(r)

# true model
velocity = np.array([-0.3, 0.2])
A_t = np.array([[1+velocity[0], 0], [0, 1+velocity[1]]])
true_1 = np.matmul(A_t, mu_0[:, None]).squeeze()
# print("true_1: ", true_1)
# Measurement model
mu_1 = np.matmul(A_t, mu_0[:, None]).squeeze() + q
# print(true_1)
# print("mu_1:" ,mu_1)
cov_1 = np.matmul(np.matmul(A_t, cov_0), A_t.T) + cov_q

# to plot measurement results
rv1 = multivariate_normal(mu_1, cov_1)
Z1 = rv1.pdf(pos)

# Observation model
radar_offset = np.array([1, 0])
H = np.array([[1, 0], [0, 0.5]])
z_1 = np.matmul(H, true_1[:, None]).squeeze() + r + radar_offset
x_1_observation = np.matmul(np.linalg.inv(H), z_1-radar_offset)

# Projected measurement in z
proj_z = np.matmul(H, mu_1) + radar_offset


# Best estimate with Kalman gain
K = np.matmul(np.matmul(cov_1, H.T), np.linalg.inv(
    np.matmul(np.matmul(H, cov_1), H.T) + cov_r))

cov_best_x = np.matmul(np.matmul(A_t, cov_0), A_t.T) + cov_q
best_x = mu_1 + np.matmul(K, (z_1 - proj_z))
rv_best = multivariate_normal(best_x, cov_best_x)
Z_best = rv_best.pdf(pos)


plt.plot(mu_0[0], mu_0[1], 'ro', markersize=3)
plt.contour(X, Y, Z, levels=2)
plt.waitforbuttonpress()

plt.plot(true_1[0], true_1[1], 'm^', markersize=10)
plt.waitforbuttonpress()


plt.arrow(mu_0[0], mu_0[1], velocity[0], velocity[1], head_width=0.1)

plt.waitforbuttonpress()

plt.plot(mu_1[0], mu_1[1], 'ro', markersize=3)
plt.contour(X, Y, Z1, levels=2)

plt.waitforbuttonpress()

plt.plot(1, 0, 'ko', markersize=20)
plt.waitforbuttonpress()
plt.plot([radar_offset[0], true_1[0]], [
         radar_offset[1], true_1[1]], 'm', linestyle="--")

plt.waitforbuttonpress()

plt.plot(z_1[0], z_1[1], 'go', markersize=5)

plt.waitforbuttonpress()

plt.plot(x_1_observation[0], x_1_observation[1], 'go', markersize=5)

plt.waitforbuttonpress()

plt.plot(best_x[0], best_x[1], 'y^', markersize=10)

plt.plot([x_1_observation[0], best_x[0]], [
         x_1_observation[1], best_x[1]], 'b', linestyle="--")
plt.plot([mu_1[0], best_x[0]], [mu_1[1], best_x[1]], 'b', linestyle="--")


plt.contour(X, Y, Z_best, levels=2)


plt.show()
