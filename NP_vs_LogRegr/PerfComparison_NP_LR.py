import numpy as np
from scipy.special import erfcinv

np.random.seed(0)

# Parameters
A = 1.0
sigma = 0.5
N = 8000            # total samples
P_FA_target = 0.05  # NP constraint

# Generate data
x_H0 = sigma * np.random.randn(N // 2)
x_H1 = A + sigma * np.random.randn(N // 2)

X = np.concatenate([x_H0, x_H1]).reshape(-1, 1)
y = np.concatenate([np.zeros(N // 2), np.ones(N // 2)])

# Shuffle once
idx = np.random.permutation(N)
X = X[idx]
y = y[idx]

# 60 / 20 / 20 split
N_train = int(0.6 * N)
N_val   = int(0.2 * N)

X_train = X[:N_train]
y_train = y[:N_train]

X_val = X[N_train:N_train + N_val]
y_val = y[N_train:N_train + N_val]

X_test = X[N_train + N_val:]
y_test = y[N_train + N_val:]

# NP threshold from theory
eta = sigma * np.sqrt(2) * erfcinv(2 * P_FA_target)

# NP Detector on TEST data
y_np_test = np_detector(X_test, eta)

# Train only on training data
w, b = train_logistic_regression(X_train, y_train)
# Logistic Regression on TEST data
probs_lr = sigmoid(w * X_test.flatten() + b)
y_lr_test = (probs_lr > 0.5).astype(int)

# Evaluate both
evaluate(y_test, y_np_test, "Neyman–Pearson Detector (Test)")
evaluate(y_test, y_lr_test, "Logistic Regression (Test)")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.5, epochs=60):
    w, b = 0.0, 0.0
    x = X.flatten()
    
    for _ in range(epochs):
        z = w * x + b
        y_hat = sigmoid(z)
        
        dw = np.mean((y_hat - y) * x)
        db = np.mean(y_hat - y)
        
        w -= lr * dw
        b -= lr * db
        
    return w, b

def np_detector(X, eta):
    return (X.flatten() > eta).astype(int)

def evaluate(y_true, y_pred, name):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    P_FA = FP / (FP + TN)
    P_D  = TP / (TP + FN)
    P_e  = (FP + FN) / len(y_true)
    
    print(f"\n{name}")
    print(f"P_FA = {P_FA:.3f}")
    print(f"P_D  = {P_D:.3f}")
    print(f"P_e  = {P_e:.3f}")
