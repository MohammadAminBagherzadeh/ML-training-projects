import numpy as np

def generate_data(n=50, noise=5.0):
    X = np.linspace(-100, 100, n)
    true_slope = 3
    true_intercept = 8
    noise = np.random.randn(n) * noise
    y = true_slope * X + true_intercept + noise
    return X, y

def h_w(x, w):
    return w[0] + w[1] * x

def cost_function(X, y, w):
    return np.sum((h_w(X, w) - y)**2) / len(X)

def gradient_descent(X, y, w, alpha, num_iters=1):
    m = len(X)
    for i in range(num_iters):
        gradient_w0 = np.sum(h_w(X, w) - y) / m
        gradient_w1 = np.sum((h_w(X, w) - y) * X) / m
        w[0] -= alpha * gradient_w0
        w[1] -= alpha * gradient_w1
    return w

def create_batches(X, y, k):
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split into k batches
    batch_size = len(X) // k
    batches = []
    for i in range(k):
        start = i * batch_size
        end = start + batch_size
        batches.append((X[start:end], y[start:end]))
    return batches

# Generate data
X, y = generate_data(n=100000, noise=5.0)

# Hyperparameters
k = 2000  # Total number of mini-batches
num_iters = 20  # Total number of iterations for each selected mini-batch
num_batches_to_use = 500  # Number of mini-batches to use in each iteration

# Learning rate
eta = 0.0005

# Initialize weights
w_initial = [0, 0]

# Mini-batches
batches = create_batches(X, y, k)

# Randomly choose 10 mini-batches to process
selected_batches = np.random.choice(len(batches), num_batches_to_use, replace=False)
    
# For each selected mini-batch, update the weights using gradient descent
for batch_idx in selected_batches:
    X_batch, y_batch = batches[batch_idx]
    w_initial = gradient_descent(X_batch, y_batch, w_initial, eta, num_iters)

# Final weights
print(f'w0 = {w_initial[0]}  w1 = {w_initial[1]}')
