from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

car_seats = pd.read_csv("CarSeats.csv")
# Removing all categorical features
car_seats.pop("ShelveLoc")
car_seats.pop("Urban")
car_seats.pop("US")
X = car_seats[["CompPrice","Income","Advertising","Population","Price" ,"Age","Education"]]
y = car_seats["Sales"]
scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)
print(f"Feature Means: {scaled_X.mean(axis=0)}")
print(f"Feature Variances: {scaled_X.var(axis=0)}")
centred_y = y.sub(y.mean())
X_train,X_test,y_train,y_test = train_test_split(scaled_X,centred_y,test_size=0.5, shuffle=False)
X_train = np.matrix(X_train)
X_test = np.matrix(X_test)
y_train = np.matrix(y_train).reshape(-1,1)
y_test = np.matrix(y_test).reshape(-1,1)
print(f"First row of X_train: {X_train[0]}")
print(f"Last row of X_train: {X_train[-1]}")
print(f"First row of X_test: {X_test[0]}")
print(f"Last row of X_test: {X_test[-1]}")
print(f"First row of y_train: {y_train[0]}")
print(f"Last row of y_train: {y_train[-1]}")
print(f"First row of y_test: {y_test[0]}")
print(f"Last row of y_test: {y_test[-1]}")

# Adding column of ones for intercept
intercept_col = np.ones((200, 1), dtype=float)
X_train = np.append(intercept_col, X_train, axis=1)
X_test = np.append(intercept_col, X_test, axis=1)

true_beta = np.linalg.inv(X_train.T @ X_train + 100 * np.identity(8)) @ X_train.T @ y_train
print(true_beta)

def L(beta):
    return 0.005 * np.linalg.norm(y_train - X_train @ beta, 2) ** 2 + 0.5 * np.linalg.norm(beta, 2) ** 2

def grad(beta):
    return 0.01 * (-X_train.T @ y_train + X_train.T @ X_train @ beta) + beta

def gen_seq(step_size):
    beta = np.array([1,1,1,1,1,1,1,1]).reshape(-1,1)
    seq = []
    for i in range(1000):
        print(beta)
        seq.append((L(beta) - L(true_beta)).item())
        beta = beta - step_size * grad(beta)
    return (seq, beta)

fig,axs = plt.subplots(3,3)
y = np.linspace(0,999,1000)

step_sizes = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

i = 0
for ax in axs.flat:
    ax.plot(gen_seq(step_sizes[i])[0])
    ax.set_title(f"\u03B1 = {step_sizes[i]}")
    ax.set(xlabel='k', ylabel='∆(k)')
    i += 1

plt.tight_layout()
plt.show()

best_beta = gen_seq(0.01)[1]
print("Batch GD MSE:")
print(f"train MSE = {(1/200) * np.linalg.norm(y_train - X_train @ best_beta, 2) ** 2}")
print(f"test MSE = {(1/200) * np.linalg.norm(y_test - X_test @ best_beta, 2) ** 2}")
print()

def sto_grad(beta, i):
    return -2 * X_train[i,].T @ y_train[i] + 2 * X_train[i,].T @ X_train[i,] * beta + beta

def gen_sto_seq(step_size):
    beta = np.array([1,1,1,1,1,1,1,1]).reshape(-1,1)
    seq = []
    for i in range(5):
        for j in range(200):
            seq.append((L(beta) - L(true_beta)).item())
            beta = beta - step_size * sto_grad(beta, j)
    return (seq, beta)

fig,axs = plt.subplots(3,3)
y = np.linspace(0,999,1000)

step_sizes = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]

i = 0
for ax in axs.flat:
    ax.plot(gen_sto_seq(step_sizes[i])[0])
    ax.set_title(f"\u03B1 = {step_sizes[i]}")
    ax.set(xlabel='k', ylabel='∆(k)')
    i += 1

plt.tight_layout()
plt.show()

best_beta = gen_sto_seq(0.006)[1]
print("SGD MSE:")
print(f"train MSE = {(1/200) * np.linalg.norm(y_train - X_train @ best_beta, 2) ** 2}")
print(f"test MSE = {(1/200) * np.linalg.norm(y_test - X_test @ best_beta, 2) ** 2}")
print()

def alt_seq():
    beta = np.array([1,1,1,1,1,1,1,1], dtype=float).reshape(-1,1)
    seq = []
    for i in range(10):
        for j in range(8):
            seq.append((L(beta) - L(true_beta)).item())
            X_no_j = np.delete(X_train, j, 1)
            beta_no_j = np.delete(beta, j, 0)
            beta[j] = (X_train[:,j].T @ y_train - X_train[:,j].T @ X_no_j @ beta_no_j) / (np.linalg.norm(X_train[:,j], 2) ** 2 + 100)
    return (seq, beta)

seq1, best_beta1 = gen_seq(0.01)
seq2, best_beta2 = gen_sto_seq(0.006)
seq3, best_beta3 = alt_seq()
fig, axs = plt.subplots()
axs.plot(seq1[:80], color='green', label='Batch GD')
axs.plot(seq2[:80], color='orange', label='SGD')
axs.plot(seq3, color='blue', label='New Algorithm')
axs.legend(loc ="upper right")
axs.set(xlabel='k', ylabel='∆(k)')
plt.show()

print("New algorithm MSE:")
print(f"train MSE = {(1/200) * np.linalg.norm(y_train - X_train @ best_beta3, 2) ** 2}")
print(f"test MSE = {(1/200) * np.linalg.norm(y_test - X_test @ best_beta3, 2) ** 2}")
print()
