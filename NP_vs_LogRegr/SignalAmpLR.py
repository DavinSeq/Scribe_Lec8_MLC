import numpy as np

np.random.seed(0)

A = 1.0
sigma = 0.5
N = 6000

x_H0 = sigma * np.random.randn(N//2)
x_H1 = A + sigma * np.random.randn(N//2)

X = np.concatenate([x_H0, x_H1]).reshape(-1, 1)
y = np.concatenate([np.zeros(N//2), np.ones(N//2)])

# Shuffle
idx = np.random.permutation(N)
X = X[idx]
y = y[idx]

# 60 / 20 / 20 split
N_train = int(0.6 * N)
N_val   = int(0.2 * N)

X_train = X[:N_train]
y_train = y[:N_train]

X_val = X[N_train:N_train+N_val]
y_val = y[N_train:N_train+N_val]

X_test = X[N_train+N_val:]
y_test = y[N_train+N_val:]

w, b = logistic_regression(X_train, y_train)

P_FA_val, P_D_val, acc_val = evaluate_classifier(X_val, y_val, w, b)

print("Validation Performance")
print(f"P_FA = {P_FA_val:.3f}")
print(f"P_D  = {P_D_val:.3f}")
print(f"Accuracy = {acc_val:.3f}")

P_FA_test, P_D_test, acc_test = evaluate_classifier(X_test, y_test, w, b)

print("\nTest Performance")
print(f"P_FA = {P_FA_test:.3f}")
print(f"P_D  = {P_D_test:.3f}")
print(f"Accuracy = {acc_test:.3f}")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.5, epochs=50):
    w = 0.0
    b = 0.0
    
    for _ in range(epochs):
        z = X.flatten() * w + b
        y_hat = sigmoid(z)
        
        # Gradients
        dw = np.mean((y_hat - y) * X.flatten())
        db = np.mean(y_hat - y)
        
        # Update
        w -= lr * dw
        b -= lr * db
        
    return w, b

def evaluate_classifier(X, y, w, b):
    probs = sigmoid(X.flatten() * w + b)
    y_pred = (probs > 0.5).astype(int)
    
    TP = np.sum((y == 1) & (y_pred == 1))
    TN = np.sum((y == 0) & (y_pred == 0))
    FP = np.sum((y == 0) & (y_pred == 1))
    FN = np.sum((y == 1) & (y_pred == 0))
    
    P_FA = FP / (FP + TN)
    P_D  = TP / (TP + FN)
    acc  = (TP + TN) / len(y)
    
    return P_FA, P_D, acc
