# -*- coding: utf-8 -*-

'''
    author: Zhewei Song
    Created on 04/08/2020
'''

import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import fmin_bfgs
import causalML.utils.tools as tools


class Propensity_Select(object):
    """
    Dictionary-like class containing propensity score data.

    Propensity score related data includes estimated logistic regression
    coefficients, maximized log-likelihood, predicted propensity scores, 
    and lists of the linear, interaction and quadratic terms that are included
    in the logistic regression.
    """

    def __init__(self, data, lin_B, C_lin, C_qua):

        X_c, X_t = data._dict['X_c'], data._dict['X_t']
        lin = select_lin_terms(X_c, X_t, lin_B, C_lin)
        qua = select_qua_terms(X_c, X_t, lin, C_qua)


def form_matrix(X, lin, qua):

    N, K = X.shape

    mat = np.empty((N, 1+len(lin)+len(qua)))
    mat[:, 0] = 1   # constant intercept

    current_col = 1  # current column number
    if lin:
        mat[:, current_col:current_col+len(lin)] = X[:, lin]
        current_col += len(lin)
    for term in qua:   # qua is a list of tuples of column numbers
        mat[:, current_col] = X[:, term[0]] * X[:, term[1]]
        current_col += 1

    return mat


def log1exp(x, top_threshold=100, bottom_threshold=-100):
    high_x = (x >= top_threshold)
    low_x = (x <= bottom_threshold)
    mid_x = ~(high_x | low_x)

    values = np.empty(x.shape[0])
    values[high_x] = 0.0
    values[low_x] = -x[low_x]
    values[mid_x] = np.log(1 + np.exp(-x[mid_x]))

    return values


def neg_loglike(beta, X_c, X_t):

    return log1exp(X_t.dot(beta)).sum() + log1exp(-X_c.dot(beta)).sum()


def calc_coef(X_c, X_t):

    K = X_c.shape[1]

    neg_ll = lambda b: neg_loglike(b, X_c, X_t)
    neg_grad = lambda b: neg_gradient(b, X_c, X_t)

    
    
def calc_loglike(X_c, X_t, lin, qua):

    Z_c = form_matirx(X_c, lin, qua)
    Z_t = form_matirx(X_t, lin, qua)
    beta = calc_coef(Z_c, Z_t)

def get_excluded_lin(K, included):

    included_set = set(included)

    return [x for x in range(K) if x not in included_set]

def select_lin(X_c, X_t, lin_B ,C_lin):

    K = X_c.shape[1]
    excluded = get_excluded_lin(K, lin_B)
    if excluded == []:
        return lin_B
    
    ll_null = calc_loglike(X_c, X_t, lin_B, [])

def select_lin_terms(X_c, X_t, lin_B, C_lin):
    if C_lin <=0:
        K = X_c.shape[1]
        return lin_B + get_excluded_lin(K, lin_B)
    elif C_lin == np.inf:
        return lin_B
    else:
        return select_lin(X_c, X_t, lin_B, C_lin)
