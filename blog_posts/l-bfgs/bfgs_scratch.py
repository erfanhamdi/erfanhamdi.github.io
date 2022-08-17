import numpy as np
import matplotlib.pyplot as plt

def central_difference_derivative(f, x, order=1):
    """
    Compute the derivative of f at x using the central difference method.
    """
    assert f.shape == x.shape
    if order == 1:
        return (f[2:] - f[:-2])/(2*(x[1]-x[0]))
    elif order == 2:
        return (f[2:]-2*f[1:-1]+f[0:-2])/(x[1]-x[0])**2

def line_search_backtrack(f, x):
    alpha = 1
    c = 0.1
    rho = 0.5
    p = -1 * central_difference_derivative(f, x, order = 1)
    while f(x + alpha * p) > f(x) + c * alpha * (-p).T @ p:
        alpha = rho * alpha
    return alpha

def calculate_hessian(f, x):
    """
    Calculate the Hessian of f at x.
    """
    return central_difference_derivative(f, x, order = 2))    
