import numpy as np

def generate_data(n=50, noise=5.0):
    X = np.linspace(-10, 10, n)
    true_slope = 3
    true_intercept = 8
    noise = np.random.randn(n) * noise
    y = true_slope * X + true_intercept + noise
    return X, y

def h_w(x, w):
    return w[0] + w[1] * x

def gradient_descent(X, y, w, alpha, num_iters=1):
    m = len(X)
    for i in range(num_iters):
        gradient_w0 = np.sum(h_w(X, w) - y) / m
        gradient_w1 = np.sum((h_w(X, w) - y) * X) / m
        w[0] -= alpha * gradient_w0
        w[1] -= alpha * gradient_w1
    return w

def cost_function(X, y, w):
    return np.sum((h_w(X, w) - y)**2) / len(X)

X, y = generate_data(n=50, noise=5.0)

eta = 0.05
num_iters = 500

w_initial = [0, 0]

w_final = gradient_descent(X, y, w_initial, eta, num_iters)

print(f"Parameters (w): ")
print(f"w_1 = {w_final[1]:.2f}, w_0 = {w_final[0]:.2f}")