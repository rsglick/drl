import numpy as np
import scipy.linalg as linalg
import gym





def lqr(A,B,Q,R):
    P = linalg.solve_continuous_are(A,B,Q,R)
    Rinv = np.linalg.inv(R)
    K = Rinv @ B.T @ P
    return K

A = np.array([])
B = np.array([])
Q = np.diag([0.1, 1, 10, 100])
R = np.array([10])
K = lqr(A, B, Q, R)

#def ulqr(x):
#    x1 = np.copy(x)
#    x1[2] = np.sin(x1[2])
#    return np.dot(K, x1)
