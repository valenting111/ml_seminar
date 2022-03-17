import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time


np.random.seed(4)
# Initial part is the same
mu_0 = np.array([4, 4])
cov_0 = 0.1*np.eye(2)
N = 400
X = np.linspace(0, 8, N)
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
r = np.random.multivariate_normal(np.array([0, 0]), cov_r)


# true model
velocity = np.array([-0.3, 0.2])
A_t = np.array([[1+velocity[0], 0], [0, 1+velocity[1]]])
true_1 = np.matmul(A_t, mu_0[:, None]).squeeze(
) + np.random.multivariate_normal(np.array([0, 0]), 0.5*np.eye(2))
# print("true_1: ", true_1)
# Measurement model
mu_1 = np.matmul(A_t, mu_0[:, None]).squeeze()
# print(true_1)
# print("mu_1:" ,mu_1)
cov_1 = np.matmul(np.matmul(A_t, cov_0), A_t.T) + cov_q

# to plot measurement results
rv1 = multivariate_normal(mu_1, cov_1)
Z1 = rv1.pdf(pos)

# This part changes

# Observation model
# still have an offset
radar_offset = np.array([0, 0])
# But
# This would be the true non linear transform
z_x = np.sqrt(true_1[0]**2+true_1[1]**2)
z_y = np.arctan(true_1[1]/true_1[0])
z_1 = np.array([z_x, z_y]) + radar_offset + r

x_1_observation_x = (z_1[0] - radar_offset[0])*np.cos(z_1[1])
x_1_observation_y = (z_1[0] - radar_offset[0])*np.sin(z_1[1])
x_1_observation = np.array([x_1_observation_x, x_1_observation_y])
# print(x_1_observation)

# But we need to approximate
f_mu_0 = np.array([np.sqrt(mu_0[0]**2+mu_0[1]**2) +
                  1, np.arctan(mu_0[1]/mu_0[0])])

# Jacobian
jacob_0_0 = mu_0[0] / np.sqrt(mu_0[0]**2 + mu_0[1]**2)
jacob_0_1 = mu_0[1] / np.sqrt(mu_0[0]**2 + mu_0[1]**2)
jacob_1_0 = (-mu_0[1]/mu_0[0]**2)*((1/np.cosh(mu_0[1]/mu_0[0]))**2)
jacob_1_1 = (1/mu_0[0])*((1/np.cosh(mu_0[1]/mu_0[0]))**2)
jacobian = np.array([[jacob_0_0, jacob_0_1], [jacob_1_0, jacob_1_1]])

jacob_mu_0 = np.matmul(jacobian, mu_0)

approx_H = jacobian
approx_b = f_mu_0 - jacob_mu_0
approx_x_1_observation = np.matmul(np.linalg.inv(approx_H), z_1-approx_b)


# Projected measurement in z
proj_z = np.matmul(approx_H, mu_1) + approx_b
true_proj_z = np.array(
    [np.sqrt(mu_1[0]**2+mu_1[1]**2), np.arctan(mu_1[1]/mu_1[0])])

# Best estimate with Kalman gain
K = np.matmul(np.matmul(cov_1, approx_H.T), np.linalg.inv(
    np.matmul(np.matmul(approx_H, cov_1), approx_H.T) + cov_r))
# print("K:",K)
correction_radar_EKF = np.matmul(K, (z_1 - proj_z))

best_x = mu_1 + correction_radar_EKF
best_cov = np.matmul(np.eye(2) - np.matmul(K, approx_H), cov_1)
best_rv1 = multivariate_normal(best_x, best_cov)
best_Z1 = best_rv1.pdf(pos)

# Unscented filtering
lower_triang = np.linalg.cholesky(cov_0)

x0 = mu_0.copy()
w0 = 1/3
x1 = mu_0 + np.sqrt(3)*lower_triang[:, 0]
w1 = 1/6
x2 = mu_0 - np.sqrt(3)*lower_triang[:, 0]
w2 = 1/6
x3 = mu_0 + np.sqrt(3)*lower_triang[:, 1]
w3 = 1/6
x4 = mu_0 - np.sqrt(3)*lower_triang[:, 1]
w4 = 1/6

weights_ukf = np.array([w0, w1, w2, w3, w4])
sigma_points = np.array([x0, x1, x2, x3, x4])

transformed_x = []
transformed_z = []

for sigma_point in sigma_points:
    transformed_x.append(np.matmul(A_t, sigma_point[:, None]).squeeze())


transformed_x = np.array([transformed_x]).squeeze()

for transf_x in transformed_x:
    transformed_z.append([np.sqrt(transf_x[0]**2+transf_x[1]
                         ** 2)+radar_offset[0], np.arctan(transf_x[1]/transf_x[0])])

transformed_z = np.array([transformed_z]).squeeze()


mean_x = np.array([0., 0.])
mean_z = np.array([0., 0.])
for idx, w_i in enumerate(weights_ukf):
    mean_x += np.multiply(w_i, transformed_x[idx])
    mean_z += np.multiply(w_i, transformed_z[idx])


cov_x = cov_q.copy()
for idx, x_i in enumerate(transformed_x):
    cov_x += weights_ukf[idx]*(np.outer(x_i - mean_x, (x_i - mean_x)))

ukf_observation = multivariate_normal(mean_x, cov_x)
UKF_observation_density = ukf_observation.pdf(pos)

mean_x_from_mean_z = np.array([(mean_z[0] - radar_offset[0])*np.cos(
    mean_z[1]), (mean_z[0] - radar_offset[0])*np.sin(mean_z[1])])

cov_z = cov_r.copy()
for idx, z_i in enumerate(transformed_z):
    cov_z += weights_ukf[idx]*(np.outer(z_i - mean_z, (z_i - mean_z)))

print("mean z: ", mean_z)

rv_cov_z_UKF = multivariate_normal(mean_z, cov_z)
Z_UKF = rv_cov_z_UKF.pdf(pos)

cov_x_z = np.array([[0., 0.], [0., 0.]])


for idx, w_i in enumerate(weights_ukf):
    cov_x_z += w_i * \
        (np.outer(transformed_x[idx]-mean_x,
         (transformed_z[idx]-mean_z)))


K_UKF = np.matmul(cov_x_z, np.linalg.inv(cov_z))
proj_z_UKF = np.array([np.sqrt(mean_x[0]**2+mean_x[1]
                               ** 2)+radar_offset[0], np.arctan(mean_x[1]/mean_x[0])])


correction_radar = np.matmul(K_UKF, (z_1 - proj_z_UKF))

mean_UKF = mean_x + correction_radar
cov_UKF = cov_x - np.matmul(np.matmul(K, cov_z), K.T)

best_rv_UKF = multivariate_normal(mean_UKF, cov_UKF)
best_Z_UKF = best_rv_UKF.pdf(pos)


plt.plot(mu_0[0], mu_0[1], 'ro', markersize=3)
plt.contour(X, Y, Z, levels=1)
plt.plot(radar_offset[0], radar_offset[1], 'ko', markersize=20)
plt.waitforbuttonpress()
plt.plot(true_1[0], true_1[1], 'm^', markersize=7)
plt.waitforbuttonpress()
plt.arrow(mu_0[0], mu_0[1], velocity[0], velocity[1], head_width=0.1)
plt.waitforbuttonpress()
plt.plot(mu_1[0], mu_1[1], 'ro', markersize=3)
plt.contour(X, Y, Z1, levels=1)
plt.waitforbuttonpress()
plt.plot([radar_offset[0], true_1[0]], [
         radar_offset[1], true_1[1]], 'm', linestyle="--")
plt.waitforbuttonpress()

plt.plot(x_1_observation[0], x_1_observation[1], 'go', markersize=5)
plt.waitforbuttonpress()


plt.plot(z_1[0], z_1[1], 'go', markersize=5)
plt.waitforbuttonpress()

plt.plot(true_proj_z[0], true_proj_z[1], 'ro', markersize=5)
plt.waitforbuttonpress()

plt.plot(proj_z[0], proj_z[1], 'ko', markersize=5)
plt.waitforbuttonpress()
plt.plot([proj_z[0], z_1[0]], [
         proj_z[1], z_1[1]], 'b', linestyle="--")

plt.waitforbuttonpress()
plt.arrow(mu_1[0], mu_1[1], correction_radar_EKF[0],
          correction_radar_EKF[1], head_width=0.1)
plt.waitforbuttonpress()


plt.plot(best_x[0], best_x[1], 'r^', markersize=7)
plt.contour(X, Y, best_Z1, levels=1)

plt.waitforbuttonpress()
plt.plot(x0[0], x0[1], 'ro', markersize=3)
plt.plot(x1[0], x1[1], 'ro', markersize=3)
plt.plot(x2[0], x2[1], 'ro', markersize=3)
plt.plot(x3[0], x3[1], 'ro', markersize=3)
plt.plot(x4[0], x4[1], 'ro', markersize=3)


plt.waitforbuttonpress()
plt.plot(transformed_x[0][0], transformed_x[0][1], 'ro', markersize=3)
plt.plot(transformed_x[1][0], transformed_x[1][1], 'ro', markersize=3)
plt.plot(transformed_x[2][0], transformed_x[2][1], 'ro', markersize=3)
plt.plot(transformed_x[3][0], transformed_x[3][1], 'ro', markersize=3)
plt.plot(transformed_x[4][0], transformed_x[4][1], 'ro', markersize=3)
plt.waitforbuttonpress()


plt.plot(mean_x[0], mean_x[1], 'ro', markersize=3)
plt.contour(X, Y, UKF_observation_density, levels=1)

plt.waitforbuttonpress()
plt.plot(transformed_z[0][0], transformed_z[0][1], 'yo', markersize=3)
plt.plot(transformed_z[1][0], transformed_z[1][1], 'yo', markersize=3)
plt.plot(transformed_z[2][0], transformed_z[2][1], 'yo', markersize=3)
plt.plot(transformed_z[3][0], transformed_z[3][1], 'yo', markersize=3)
plt.plot(transformed_z[4][0], transformed_z[4][1], 'yo', markersize=3)

plt.plot(mean_z[0], mean_z[1], 'bo', markersize=5)
plt.contour(X, Y, Z_UKF, levels=1)
plt.waitforbuttonpress()
plt.plot([mean_z[0], z_1[0]], [
         mean_z[1], z_1[1]], 'b', linestyle="--")


plt.waitforbuttonpress()
plt.arrow(mean_x[0], mean_x[1], correction_radar[0],
          correction_radar[1], head_width=0.1)

plt.waitforbuttonpress()
# plt.plot([mean_UKF[0], correction_radar[0]], [
#          mean_UKF[1], correction_radar[1]], 'b', linestyle="--")
plt.plot([mean_UKF[0], mean_x[0]], [
         mean_UKF[1], mean_x[1]], 'b', linestyle="--")

plt.plot(mean_UKF[0], mean_UKF[1], 'y^', markersize=7)
plt.contour(X, Y, best_Z_UKF, levels=1)
plt.show(block=True)


# plt.plot(mu_0[0], mu_0[1], 'ro', markersize=3)
# plt.contour(X, Y, Z, levels=1)
# plt.plot(radar_offset[0], radar_offset[1], 'ko', markersize=20)
# # plt.waitforbuttonpress()
# plt.plot(true_1[0], true_1[1], 'm^', markersize=7)
# # plt.waitforbuttonpress()
# plt.plot([radar_offset[0], true_1[0]], [
#          radar_offset[1], true_1[1]], 'm', linestyle="--")
# # plt.waitforbuttonpress()
# plt.arrow(mu_0[0], mu_0[1], velocity[0], velocity[1], head_width=0.1)
# # plt.waitforbuttonpress()

# plt.plot(z_1[0], z_1[1], 'go', markersize=5)
# # plt.waitforbuttonpress()

# plt.plot(x_1_observation[0], x_1_observation[1], 'go', markersize=5)
# plt.waitforbuttonpress()

# plt.plot(x0[0], x0[1], 'ro', markersize=3)
# plt.plot(x1[0], x1[1], 'ro', markersize=3)
# plt.plot(x2[0], x2[1], 'ro', markersize=3)
# plt.plot(x3[0], x3[1], 'ro', markersize=3)
# plt.plot(x4[0], x4[1], 'ro', markersize=3)


# plt.waitforbuttonpress()
# plt.plot(transformed_x[0][0], transformed_x[0][1], 'ro', markersize=3)
# plt.plot(transformed_x[1][0], transformed_x[1][1], 'ro', markersize=3)
# plt.plot(transformed_x[2][0], transformed_x[2][1], 'ro', markersize=3)
# plt.plot(transformed_x[3][0], transformed_x[3][1], 'ro', markersize=3)
# plt.plot(transformed_x[4][0], transformed_x[4][1], 'ro', markersize=3)
# plt.waitforbuttonpress()


# plt.plot(mean_x[0], mean_x[1], 'ro', markersize=3)
# plt.contour(X, Y, UKF_observation_density, levels=1)

# plt.waitforbuttonpress()
# plt.plot(transformed_z[0][0], transformed_z[0][1], 'yo', markersize=3)
# plt.plot(transformed_z[1][0], transformed_z[1][1], 'yo', markersize=3)
# plt.plot(transformed_z[2][0], transformed_z[2][1], 'yo', markersize=3)
# plt.plot(transformed_z[3][0], transformed_z[3][1], 'yo', markersize=3)
# plt.plot(transformed_z[4][0], transformed_z[4][1], 'yo', markersize=3)

# plt.waitforbuttonpress()
# plt.plot(mean_z[0], mean_z[1], 'bo', markersize=5)
# plt.plot([mean_z[0], z_1[0]], [
#          mean_z[1], z_1[1]], 'b', linestyle="--")


# plt.waitforbuttonpress()
# plt.arrow(mean_x[0], mean_x[1], correction_radar[0],
#           correction_radar[1], head_width=0.1)

# plt.waitforbuttonpress()
# # plt.plot([mean_UKF[0], correction_radar[0]], [
# #          mean_UKF[1], correction_radar[1]], 'b', linestyle="--")
# plt.plot([mean_UKF[0], mean_x[0]], [
#          mean_UKF[1], mean_x[1]], 'b', linestyle="--")

# plt.plot(mean_UKF[0], mean_UKF[1], 'y^', markersize=7)
# plt.contour(X, Y, best_Z_UKF, levels=1)
# plt.show(block=True)
