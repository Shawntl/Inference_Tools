# -*- coding: utf-8 -*-

'''
    author: Zhewei Song
    Created on 03/02/2020
'''


import numpy as np 


class Data(object):
    """
    Dictionary-like class containing basic data.
    """

    def __init__(self, outcome, treatment, covariates):

        Y, T, X = preprocess(outcome, treatment, covariates)
        self._dict = dict()
        self._dict['Y'] = Y
        self._dict['T'] = T
        self._dict['X'] = X
        self._dict['N'], self._dict['K'] = X.shape
        self._dict['Y_c'], self._dict['Y_t'] = Y[T == 0], Y[T == 1]
        self._dict['X_c'], self._dict['X_t'] = X[T == 0], X[T == 1]
        self._dict['N_t'] = T.sum()
        self._dict['N_c'] = self._dict['N'] - self._dict['N_t']
        if self._dict['K']+1 > self._dict['N_c']:
            raise ValueError('Too few control units: N_c < K+1')
        if self._dict['K']+1 > self._dict['N_t']:
            raise ValueError('Too few treated units: N_t < K+1')


def preprocess(Y, T, X):
    Y = Y.values
    T = T.values
    X = X.values
    if Y.shape[0] == T.shape[0] == X.shape[0]:
        N = Y.shape[0]
    else:
        raise IndexError('Input data have different number of rows')

    if Y.shape != (N, ):
        Y.shape = (N, )
    if T.shape != (N, ):
        T.shape = (N, )
    if T.dtype != 'int':
        T = T.astype(int)
    if X.shape == (N, ):
        X.shape = (N, 1)

    return (Y, T, X)
