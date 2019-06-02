#############################################
# Created on Mon June 3 2019                #
# Author: Youngjoon Yoon                    #
# Convex Optimization Programming Homework  #
# Spring 2019                               #
#############################################
# Algorithm for Gradient Descent Method     
# fun: function, grad: gradient of fun
import numpy as np

def min_gd(fun, x0, grad, args=()):

    epsilon = 10**(-5)
    x = x0
    alpha = 0.3
    beta = 0.8
    t = 1

    while np.linalg.norm(grad(x, *args)) > epsilon:

        dx = -grad(x, *args)

        while fun(x + t*dx, *args) > fun(x, *args) + alpha * t * grad(x, *args).T@dx :

            t = beta * t

        x = x + t * dx


    return x
