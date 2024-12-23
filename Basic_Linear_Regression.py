import numpy as np

def generate_data(n, noise):
    x = np.linspace(-10, 10, n)
    true_slope = 3
    true_intercept = 8
    noise = np.random.randn(n) * noise
    y = true_slope * x + true_intercept + noise
    return x, y

x, y = generate_data(n=10, noise=5.0)

X, Y, XiYi, Xi2 = 0, 0, 0, 0

n = len(x)
for i in range(n):
    X = X+x[i]
    Y = Y+y[i]
    XiYi = XiYi+(x[i]*y[i])
    Xi2 = Xi2+(x[i]**2)

w1 = ((n*XiYi)-(X*Y))/((n*Xi2)-(X**2))
w0 = (Y-(w1*X))/n

print(f'w1 = {w1} , w0 = {w0}')





