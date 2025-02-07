import numpy as np


def generate_data(n=50, noise=5.0):
    X = np.linspace(-10, 10, n)
    true_slope = 3
    true_intercept = 8
    noise = np.random.randn(n) * noise
    y = true_slope * X + true_intercept + noise
    return X, y

def linear_regression_closed_form(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return w

def h_w(x, w):
    return w[0] + w[1] * x

X, y = generate_data(n=500, noise=5.0)

w = linear_regression_closed_form(X, y)
print(f"Parameters (w): ")
print(f"w_1 = {w[1]:.2f}, w_0 = {w[0]:.2f}")
