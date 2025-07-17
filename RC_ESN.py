import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp

# Class of ESN (for a single task)
#============================== ESN ====================================
class ESN():
    def __init__(self,N,rho,sigma,p,pin,rseed=0):
        self.N,self.rho,self.sigma,self.p,self.pin = N,rho,sigma,p,pin
        np.random.seed(rseed)
        W = np.random.uniform(-1,1,(N,N))*(np.random.uniform(0,1,(N,N))<p)
        eigs = np.linalg.eigvals(W)
        self.W = rho*W/np.max(np.abs(eigs))
        np.random.seed(rseed+1)
        self.Win = np.random.uniform(-sigma,sigma,N)*(np.random.uniform(0,1,N)<pin)

    def run(self,u,f=np.tanh,X0=[]):
        if len(X0)==0:
            X0 = np.ones(self.N)
        T = len(u)
        X = np.zeros((T,self.N))
        X[0] = X0
        for t in range(1,T):
            X[t] = f(self.W @ X[t-1] + self.Win*u[t-1])
        return X

    def washout(self,X,y,Two):
        T = X.shape[0]-Two
        Xwo = np.hstack([X[Two:],np.ones((T,1))])
        ywo = y[Two:].reshape((T,1))
        return Xwo,ywo

    def linear_regression(self,Xwo,ywo,Ttrain):
        # Split time-series into training and evaluation phases
        Xtrain, Xeval = Xwo[:Ttrain], Xwo[Ttrain:]
        ytrain, yeval = ywo[:Ttrain], ywo[Ttrain:]
        # Calculate wout and output
        wout = np.linalg.pinv(Xtrain) @ ytrain
        yhat = Xeval @ wout
        # NRMSE
        NRMSE = np.sqrt(np.mean((yhat - yeval) ** 2))/np.std(yeval)
        return yeval,yhat,NRMSE

# #============================ TARGET ===================================
def narma10(v,a=0.3,b=0.05,c=1.5,d=0.1):
    T = len(v)
    y = np.zeros(T)
    for t in range(10,T):
        y[t] = a * y[t-1] + b * y[t-1] * (np.sum(y[t-10:t])) + c * v[t-10] * v[t-1] + d
    return y


def nonlinear_rec_10(v, a=0.4, b=0.1, c=1.2, d=0.05, e=0.3):
    T = len(v)
    y = np.zeros(T)
    
    for t in range(10, T):
        y[t] = (
            a * y[t-1] 
             + b * y[t-1] #  np.sum(y[t-10:t])  # Terme quadratique mémoire
            + c * np.tanh(v[t-10] * v[t-1])  # Non-linéarité sigmoïde-like
            + d * np.cos(y[t-5])  # Interaction oscillatoire
            + e
        )
    
    return y

#======================= PARAMETERS =======================

N=100
rho= 0.5
sigma=0.4
p, pin=1,1
Two = 150
Ttrain, Teval = 2000, 1000
T = Ttrain+Teval
np.random.seed(0)
u = np.random.uniform(0, 1, Two + T)

v = 0.1 * (u + 1)
y= narma10(v,a=0.3,b=0.05,c=1.5,d=0.1)
print(f"Value of y(t) = {y[0:10]}")
#============================== RUN_THE_ESN ======================
esn = ESN(N,rho,sigma,p,pin)
X= esn.run(u,f=np.tanh,X0=[])
Xwo,ywo=esn.washout(X,y,Two)
yeval,yhat,NRMSE = esn.linear_regression(Xwo,ywo,Ttrain)

# # ================GRILL DE SEARCH =====================

# rhos = np.arange(0.1,1.51,0.1)
# mus = np.arange(0.5,10.01,0.5)
# sigmas = np.arange(0.1,0.51,0.1)
# p,pin = 1,1

# # Input
# # np.random.seed(0)
# u_asym = np.random.uniform(0,1,Two+T) # u in [0,1]
# u_sym = 2*u_asym-1 # u in [-1,1]

# # Run NARMA10
# v = 0.15*u_asym # v in [0.0,0.5]
# y = narma10()
# nrmses = np.zeros((len(rhos),len(mus),len(sigmas)))

# for i,rho in enumerate(rhos):
#     for j,mu in enumerate(mus):
#         for k,sigma in enumerate(sigmas):
#             # Biased and scaled input
#             u = mu + sigma*u_sym
#             # Run ESN
#             esn = ESN(N,rho,sigma,p,pin)
#             X = esn.run(u)
#             Xwo,ywo = esn.washout(X,y,Two)
#             yeval,yhat,nrmse = esn.linear_regression(Xwo,ywo,Ttrain)
#             print(i,j,k,'rho',rho,'mu',mu,'sigma',sigma,'NRMSE',nrmse)
#             nrmses[i,j,k] = nrmse

# index = np.unravel_index(np.argmin(nrmses),nrmses.shape)
# print(index)
# rho_opt,mu_opt,sigma_opt = rhos[index[0]],mus[index[1]],sigmas[index[2]]
# print('NRMSE',np.min(nrmses))
# print('rho',rho_opt,'mu',mu_opt,'sigma',sigma_opt)


# ================================== RESULTS ===================
if NRMSE<0.2:
    print(f"Value of NRMSE = {NRMSE} < 0.2 --> GOOD PERFORMANCE")
else:
    print(f"Value of NRMSE = {NRMSE} > 0.2 --> BAD PERFORMANCE, can be improve")

# ========================== CHECK THE PHASES WASHOUT / TRAINING / EVAL =================

plt.figure(figsize=(16, 5))

# Subplot 1 : Courbes temporelles des variables
plt.subplot(1, 3, 1)
plt.plot(X[:Two, 0], label=r'$x_1(t)$')
plt.plot(X[:Two, 1], label=r'$x_2(t)$')
plt.plot(X[:Two, 2], label=r'$x_3(t)$')
plt.xlabel('Tiomestep')
plt.ylabel(r'$x_{i}(t)$')
plt.legend(loc="upper right", frameon=False)

# Subplot 2 : Trajectoire 3D (phase de washout)
ax = plt.subplot(1, 3, 2, projection='3d')
ax.plot(X[:Two, 0], X[:Two, 1], X[:Two, 2], color="black", linewidth=1)
ax.set_xlabel(r'$x_1(t)$')
ax.set_ylabel(r'$x_2(t)$')
ax.set_zlabel(r'$x_3(t)$')
ax.set_title(r'Washout phase ($t=(1-T_{\rm washout}),\ldots,0$)')

# Subplot 3 : Trajectoire 3D (entraînement + évaluation)
ax = plt.subplot(1, 3, 3, projection='3d')
ax.plot(X[Two:, 0], X[Two:, 1], X[Two:, 2], color="black", linewidth=1)
ax.set_xlabel(r'$x_1(t)$')
ax.set_ylabel(r'$x_2(t)$')
ax.set_zlabel(r'$x_3(t)$')
ax.set_title(r'Training and evaluation phases ($t=1,\ldots,T_{\rm train}+T_{\rm eval}$)')

plt.tight_layout()
plt.show()


# ======================PLOT TARGET VS OUTPUT ===========================

fig = plt.figure(figsize=(8,5))
plt.plot(yeval[:100,0],label=r'Target $y_t$')
plt.plot(yhat[:100,0],label=r'Output $\hat{y}_t$')
plt.xlabel('Timestep')
plt.ylabel(r'$y_t, \hat{y}_t$')
plt.legend(loc='lower right',frameon=False)
plt.show()



