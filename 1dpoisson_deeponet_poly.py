import torch
import numpy as np
import deepxde as dde
import neuralop
from neuralop.models import FNO
import matplotlib.pyplot as plt
import random

# Poisson equation: -u_xx = f
def equation(x, y, f):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - f


# Domain is interval [0, 1]
geom = dde.geometry.Interval(0, 1)


# Zero Dirichlet BC
def u_boundary(_):
    return 0


def boundary(_, on_boundary):
    return on_boundary


bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)

# Define PDE
pde = dde.data.PDE(geom, equation, bc, num_domain=100, num_boundary=2)

# Function space for f(x) are polynomials
degree = 3
space = dde.data.PowerSeries(N=degree + 1)

# Choose evaluation points
num_eval_points = 10
evaluation_points = geom.uniform_points(num_eval_points, boundary=True)

# Define PDE operator
pde_op = dde.data.PDEOperatorCartesianProd(
    pde,
    space,
    evaluation_points,
    num_function=100,
)

# Setup DeepONet
dim_x = 1
p = 32
net = dde.nn.DeepONetCartesianProd(
    [num_eval_points, 32, p],
    [dim_x, 32, p],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

# Define and train model
model = dde.Model(pde_op, net)
dde.optimizers.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

# Plot realisations of f(x)
n = 5
features = space.random(n)
fx = space.eval_batch(features, evaluation_points)

x = geom.uniform_points(100, boundary=True)
y = model.predict((fx, x))

# Setup figure
fig = plt.figure(figsize=(7, 8))
plt.subplot(2, 1, 1)
plt.title("Poisson equation: Source term f(x) and solution u(x)")
plt.ylabel("f(x)")
z = np.zeros_like(x)
plt.plot(x, z, "k-", alpha=0.1)

# Plot source term f(x)
for i in range(n):
    plt.plot(evaluation_points, fx[i], "--")

# Plot solution u(x)
plt.subplot(2, 1, 2)
plt.ylabel("u(x)")
plt.plot(x, z, "k-", alpha=0.1)
for i in range(n):
    plt.plot(x, y[i], "-")
plt.xlabel("x")

plt.show()

########error plots##########

u_g = np.zeros_like(y)
for i in range(n):
    # Get polynomial coefficients in descending order of degree
    coeffs = features[i][::-1]
    # Define polynomial f_i(x)
    p = np.poly1d(coeffs)
    # Integrate twice to get -p_int2(x)
    p_int2 = p.integ(m=2)
    # Apply boundary conditions to solve for constants C0 and C1
    C0 = 0  # Since u(0) = 0
    C1 = (p_int2(1) - p_int2(0)) / 1  # Since u(1) = 0
    # Compute ground truth solution u_{g_i}(x)
    u_g_i_x = -p_int2(x) + C1 * x + C0
    u_g[i] = u_g_i_x.squeeze()  # Ensure u_g_i_x is one-dimensional

# Compute errors e_i(x)
errors = y.squeeze() - u_g  # Ensure y is also one-dimensional
x = geom.uniform_points(100, boundary=True)
# Plot errors
fig, axes = plt.subplots(n, 1, figsize=(7, 2 * n))
for i in range(n):
    axes[i].plot(x, errors[i])
    axes[i].set_title(f"Error $e_{i+1}(x) = u_{i+1}(x) - u_{{g_{i+1}}}(x)$")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("Error")
plt.tight_layout()
plt.show()