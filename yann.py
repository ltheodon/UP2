#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:08:11 2021

@author: yann
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import poisson


def Poisson_Process(l, x_size, y_size):
    # Conditional Poisson Point Process
    # uniform distribution
    # nb_points: number of points
    nb_points = poisson(l*x_size*y_size)
    # xmin, xmax, ymin, ymax: defined the domain (window)
    x = (x_size)*np.random.rand(nb_points)
    y = (y_size)*np.random.rand(nb_points)
    return x, y


from scipy.spatial.distance import cdist, pdist


def Count_Pairs(x, y, R=0.2):
    """
    Ripley K and L functions, vals is values of radius
    this function has border effects!
    x, y: coordinates of points
    xmin, xmax, ymin, ymax: window
    edges: values of bins for histogram evaluation
    """

    # compute pairwise distances
    P = np.transpose(np.vstack((x, y)))
    d = pdist(P)

    return np.count_nonzero(d < R)


x, y = Poisson_Process(100, 1, 1)


def normal_distribution(nb_points, mu, sigma):
    # Normal distribution centered around the point mu with stdev sigma
    x = mu[0] + sigma[0]*np.random.randn(nb_points)
    y = mu[1] + sigma[1]*np.random.randn(nb_points)
    return x, y

def Strauss(n, gamma, R, N):

    #np.random.seed(0)
    x_max=1
    y_max=1
    
    X = x_max * np.random.rand(n)
    Y = y_max * np.random.rand(n)
    X2 = X.copy()
    Y2 = Y.copy()
    
    for i in range(N):
        beta = np.random.rand()
        idx = np.random.randint(0, n)
        dpl_x, dpl_y = normal_distribution(1, [0, 0], [.1, .1])
        
        x = X[idx] + dpl_x
        y = Y[idx] + dpl_y
        
        if x<0 or x > x_max or y <0 or y>y_max:
            alpha=0
            continue
           
        X2 = X.copy()
        Y2 = Y.copy()
        X2[idx] = x
        Y2[idx] = y
        d = Count_Pairs(X2, Y2, R) - Count_Pairs(X, Y, R)
        if d <= 0:
            alpha = 1
            
        else:
            alpha = gamma**d
            
        if beta < alpha:
            X = X2.copy()
            Y = Y2.copy()
      
    return X, Y

X, Y = Strauss(100, .1, .2, 10000)
plt.plot(X, Y, '+')
plt.show()

X, Y = Strauss(100, .1, 0.05, 10000)
plt.plot(X, Y, '+')
plt.show()

X, Y = Strauss(100, 1, .5, 10000)
plt.plot(X, Y, '+')
plt.show()

#%%
import skimage.measure
import skimage.io

def minkowski_functionals(X):
    # X: binary set
    
    return np.sum(X), skimage.measure.perimeter(X), skimage.measure.euler_number(X, connectivity=0)

from glob import glob

listOfFiles = glob("../pix20/*.png")
W = np.zeros((len(listOfFiles), 3))
for i, filename in enumerate(listOfFiles):
    X = skimage.io.imread(filename)
    W[i, :] = minkowski_functionals(X>100)

W = np.mean(W, axis=0)

def f(val):
    # comparison with
    rMean = np.mean(val[0])
    areaMean = np.pi*rMean**2
    perMean = 2*np.pi*rMean

    W_X = val[1] * np.array([areaMean, perMean/2, np.pi])
    W_0 = 1-np.exp(-W_X[0])
    W_1 = np.exp(-W_X[0]) * W_X[1]
    W_2 = np.exp(-W_X[0]) * (W_X[2] - W_X[1]**2)

    error_0 = np.abs(W_0-W[0]) / W_0
    error_1 = np.abs(W_1-W[1]) / W_1
    error_2 = np.abs(W_2-W[2]) / W_2
    
    return error_0+error_1+error_2

f([200,200])
from scipy.optimize import fmin
opt = fmin(f, np.array([.001, 200]))