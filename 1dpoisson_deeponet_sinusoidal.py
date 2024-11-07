import torch
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

# Set default dtype for PyTorch
torch.set_default_dtype(torch.float32)

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

# Function space for f(x) are sinusoidal functions
class SinusoidalSpace:
    def __init__(self, num_terms):
        self.num_terms = num_terms

    def random(self, n):
        # Generate random coefficients for sin and cos terms
        return np.random.randn(n, 2 * self.num_terms).astype(np.float32)

    def eval_batch(self, features, x):
        batch_size = features.shape[0]
        x = x.reshape(-1, 1).astype(np.float32)
        result = np.zeros((batch_size, x.shape[0]), dtype=np.float32)
        for i in range(self.num_terms):
            result += features[:, 2*i:2*i+1] * np.sin((i+1) * np.pi * x.T) + \
                      features[:, 2*i+1:2*i+2] * np.cos((i+1) * np.pi * x.T)
        return result

# Choose number of sinusoidal terms
num_terms = 5
space = SinusoidalSpace(num_terms)

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
# model.compile("adam", lr=0.001)
# losshistory, train_state = model.train(iterations=1000)
dde.optimizers.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

# Plot realizations of f(x)
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

# Error plots
u_g = np.zeros_like(y)
for i in range(n):
    # Compute ground truth solution u_{g_i}(x)
    u_g_i_x = np.zeros_like(x, dtype=np.float32)
    for j in range(num_terms):
        a_j = features[i, 2*j]
        b_j = features[i, 2*j+1]
        k = j + 1
        u_p = (a_j * np.sin(k * np.pi * x) + b_j * np.cos(k * np.pi * x)) / (k * np.pi)**2
        # Add homogeneous solution to satisfy boundary conditions
        u_h = -u_p[0] * (1 - x) - u_p[-1] * x
        u_g_i_x += u_p + u_h
    u_g[i] = u_g_i_x.squeeze()


# Compute errors e_i(x)
errors = y.squeeze() - u_g
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