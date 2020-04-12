# -*- coding: utf-8 -*-

'''
    author: Zhewei Song
    Created on 03/07/2020
'''

import numpy as np  
import pandas as pd
from .instance import OutFormat
from functools import reduce


class Matching(OutFormat):

    def __init__(self, data, covariate_name, W, m, bias_adj):

        self._method = "Matching"
        self.covariate_name = covariate_name
        self.N, self.N_c, self.N_t = data._dict['N'], data._dict['N_c'], data._dict['N_t']
        self.Y_c, self.Y_t = data._dict['Y_c'], data._dict['Y_t']
        self.X_c, self.X_t = data._dict['X_c'], data._dict['X_t']

        # match counterfactual pairs for controal group and
        # treatment group.
        self.matches_c = [match(X_i, self.X_t, W, m) for X_i in self.X_c]
        self.matches_t = [match(X_i, self.X_c, W, m) for X_i in self.X_t]

        # calculate potential outcome for each group
        Y_potent_c = np.array([self.Y_t[idx].mean() for idx in self.matches_c])
        Y_potent_t = np.array([self.Y_c[idx].mean() for idx in self.matches_t])
        ITT_c = Y_potent_c - self.Y_c
        ITT_t = self.Y_t - Y_potent_t

        if bias_adj:
            bias_coefs_c = bias_coefs(self.matches_c, self.Y_t, self.X_t)
            bias_coefs_t = bias_coefs(self.matches_t, self.Y_c, self.X_c)
            bias_c = bias(self.X_c, self.X_t, self.matches_c, bias_coefs_c)
            bias_t = bias(self.X_t, self.X_c, self.matches_t, bias_coefs_t)
            ITT_c = ITT_c - bias_c
            ITT_t = ITT_t + bias_t
            
        self._dict = dict()
        self._dict['atc'] = ITT_c.mean()
        self._dict['att'] = ITT_t.mean()
        self._dict['ate'] = (self.N_t/self.N)*self._dict['att'] + (self.N_c/self.N)*self._dict['atc']

        scaled_counts_c = scaled_counts(self.N_c, self.matches_t)
        scaled_counts_t = scaled_counts(self.N_t, self.matches_c)
        vars_c = np.repeat(ITT_c.var(), self.N_c)
        vars_t = np.repeat(ITT_t.var(), self.N_t)

        # a kind of standard error
        self._dict['atc_se'] = calc_atc_se(vars_c, vars_t, scaled_counts_t)
        self._dict['att_se'] = calc_att_se(vars_c, vars_t, scaled_counts_c)
        self._dict['ate_se'] = calc_ate_se(vars_c, vars_t, 
                                           scaled_counts_c,
                                           scaled_counts_t)

    def Get_Data(self, effect_type='att'):

        if effect_type == 'att':
            match_arr = np.concatenate((self.X_t, 
                                        self.Y_t.reshape(self.Y_t.shape[0], 1),
                                        np.ones((self.Y_t.shape[0], 1), dtype='int')), axis=1)
            match_idx = self.matches_t
            match_X = self.X_c
            match_Y = self.Y_c
        else:
            match_arr = np.concatenate((self.X_c, 
                                        self.Y_c.reshape(self.Y_c.shape[0], 1),
                                        np.zeros((self.Y_c.shape[0], 1), dtype='int')), axis=1)
            match_idx = self.matches_c
            match_X = self.X_t
            match_Y = self.Y_t
            match_T = np.ones((match_Y.shape[0], 1), dtype='int')

        for idx in match_idx:
            if effect_type == 'att':
                match_T = np.zeros((match_Y[idx].shape[0], 1), dtype='int')
            else:
                match_T = np.ones((match_Y[idx].shape[0], 1), dtype='int')
            new_arr = np.concatenate((match_X[idx], 
                                      match_Y[idx].reshape(match_Y[idx].shape[0], 1), 
                                      match_T), axis=1)

            match_arr = np.concatenate((match_arr, new_arr), axis=0)

        match_data = pd.DataFrame(data=match_arr, 
                                  columns=self.covariate_name+['PIC50']+['treat'])

        return match_data


def norm(X_i, X_m, W):

    dX = X_m - X_i
    if W.ndim == 1:
        return (dX**2 * W).sum(1)
    else:
        return (dX.dot(W) * dX).sum(1)


def smallestm(d, m):
    # Find indices of the smallest m numbers in an array.

    par_idx = np.argpartition(d, m)

    return par_idx[:m+1]


def match(X_i, X_m, W, m):

    d = norm(X_i, X_m, W)

    return smallestm(d,m)


def bias_coefs(matches, Y_m, X_m):
    # Linear model parameter which used for potential outcome calculation

    flat_idx = reduce(lambda x, y: np.concatenate((x, y)), matches)
    N, K = len(flat_idx), X_m.shape[1]

    Y = Y_m[flat_idx]
    X = np.empty((N, K+1))
    X[:, 0] = 1  # intercept term
    X[:, 1:] = X_m[flat_idx]

    return np.linalg.lstsq(X, Y)[0][1:]  # don't need intercept coef


def bias(X, X_m, matches, coefs):
    
    # Computes bias correction term, which is approximated by the dot
    # product of the matching discrepancy (i.e, X-X matched) and the 
    # coefficients from the bias correction regression.

    X_m_mean = [X_m[idx].mean(0) for idx in matches]
    bias_list = [(X_j - X_i).dot(coefs) for X_i, X_j in zip(X, X_m_mean)]

    return np.array(bias_list)


def scaled_counts(N, matches):

    # Counts the number of times each subject has appeared as a match. In 
    # the case of multiple matches, each subject only gets partial credit.

    s_counts = np.zeros(N)

    for matches_i in matches:
        scale = 1 / len(matches_i)
        for match in matches_i:
            s_counts[match] += scale

    return s_counts


def calc_atx_var(vars_c, vars_t, weights_c, weights_t):

    N_c, N_t = len(vars_c), len(vars_t)
    summands_c = weights_c**2 * vars_c
    summands_t = weights_t**2 * vars_t

    return summands_t.sum()/N_t**2 + summands_c.sum()/N_c**2


def calc_atc_se(vars_c, vars_t, scaled_counts_t):

    N_c, N_t = len(vars_c), len(vars_t)
    weights_c = np.ones(N_c)
    weights_t = (N_t/N_c) * scaled_counts_t

    var = calc_atx_var(vars_c, vars_t, weights_c, weights_t)

    return np.sqrt(var)


def calc_att_se(vars_c, vars_t, scaled_counts_c):

    N_c, N_t = len(vars_c), len(vars_t)
    weights_c = (N_c/N_t) * scaled_counts_c
    weights_t = np.ones(N_t)

    var = calc_atx_var(vars_c, vars_t, weights_c, weights_t)

    return np.sqrt(var)


def calc_ate_se(vars_c, vars_t, scaled_counts_c, scaled_counts_t):

    N_c, N_t = len(vars_c), len(vars_t)
    N = N_c + N_t
    weights_c = (N_c/N)*(1+scaled_counts_c)
    weights_t = (N_t/N)*(1+scaled_counts_t)

    var = calc_atx_var(vars_c, vars_t, weights_c, weights_t)

    return np.sqrt(var)

