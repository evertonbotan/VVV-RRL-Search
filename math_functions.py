'''
Functions to Fit

'''
import numpy as np

def linear(x,a,b):
    y = a*x+b
    return y

def gaussian(x, a1, b1, c1):#, a2, b2, c2):#, a3, b3, c3):
    y = a1 * np.exp(-(x-b1)**2 / (2*c1**2)) #+ a2 * np.exp(-(x-b2)**2 / (2*c2**2))# + a3*x**2 + b3*x +c3
    return y

def single_gaussian(x,params):
    c1, mu1, sigma1 = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) )
    return res

def single_gaussian_residuals(params,x,y):
    res = y - single_gaussian(x, params)
    return res

def double_gaussian(x,params):
    c1, mu1, sigma1, c2, mu2, sigma2 = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

def double_gaussian_residuals(params,x,y):
    res = y - double_gaussian(x, params)
    return res

def exponential(x,params):
    a,b,c = params
    res = a + b*np.exp(c*x)
    return res

def exponential_residuals(params,x,y):
    res = y - exponential(x, params)
    return res