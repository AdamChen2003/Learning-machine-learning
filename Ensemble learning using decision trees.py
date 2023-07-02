from sklearn.tree import DecisionTreeRegressor
import numpy as np 
import matplotlib.pyplot as plt

# true function
def f(x):
    t1 = np.sqrt(x * (1-x))
    t2 = (2.1 * np.pi) / (x + 0.05)
    t3 = np.sin(t2)
    return t1*t3

def f_sampler(f, n=100, sigma=0.05):    
    # sample points from function f with Gaussian noise (0,sigma**2)
    xvals = np.random.uniform(low=0, high=1, size=n)
    yvals = f(xvals) + sigma * np.random.normal(0,1,size=n)

    return xvals, yvals

np.random.seed(123)
X, y = f_sampler(f, 160, sigma=0.2)
X = X.reshape(-1,1)

fig = plt.figure(figsize=(7,7))
dt = DecisionTreeRegressor(max_depth=1).fit(X,y)
xx = np.linspace(0,1,1000)

def predict(T, depth, alpha):
    n = np.shape(X)[0]
    ft = np.zeros(np.shape(y)[0])
    fT = np.zeros(np.shape(xx)[0])
    for t in range(T):
        r = y - ft
        ht = DecisionTreeRegressor(max_depth=depth).fit(X,r)
        if alpha is None:
            alpha = ((y - ft).T @ ht.predict(X))/np.linalg.norm(ht.predict(X), 2)**2
        ft += alpha * ht.predict(X)
        fT += alpha * ht.predict(xx.reshape(-1,1))

    return fT

depth = 2

# With adaptive step-size
fig, axs = plt.subplots(5,2)
T_list = [5,10,15,20,25,30,35,40,45,50]

for i, ax in enumerate(axs.flat):
    ax.plot(xx, f(xx), alpha=0.5, color='red', label='truth')
    ax.plot(xx, predict(T_list[i], depth, None), color='blue', label='predicted')
    ax.set_title(f"base learners: {T_list[i]}")

plt.tight_layout()
plt.show()

# With fixed step-size = 0.1
fig, axs = plt.subplots(5,2)
T_list = [5,10,15,20,25,30,35,40,45,50]

for i, ax in enumerate(axs.flat):
    ax.plot(xx, f(xx), alpha=0.5, color='red', label='truth')
    ax.plot(xx, predict(T_list[i], depth, 0.1), color='blue', label='predicted')
    ax.set_title(f"base learners: {T_list[i]}")

plt.tight_layout()
plt.show()

    
