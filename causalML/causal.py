# -*- coding: utf-8 -*-

'''
    author: Zhewei Song
    Created on 03/02/2020
'''


from .preprocess import Data, Summary
from .estimators import Estimators, Matching
from itertools import combinations_with_replacement
import numpy as np


class CausalMethod(object):
    '''
    Class that provides the main tools of Causal Inference.
    '''

    def __init__(self, Y, T, X):
        self.history_data = Data(Y, T, X)
        self.covariate_name = list(X.columns)
        self.reset()

    def reset(self):
        """
        Reinitializes data to original inputs.
        """
        Y, T, X = self.history_data._dict['Y'], self.history_data._dict['T'], self.history_data._dict['X']
        self.raw_data = self.history_data
        self.summary_stats = Summary(self.raw_data, self.covariate_name)
        self.estimates = Estimators()
        self.match_att_df = None
        self.match_atc_df = None
        self.att_match_stats = None
        self.atc_match_stats = None
        self.propensity = None

    def est_via_matching(self, weights='inv', matches=1, bias_adj=False):
        """
        Estimates averge treatment effects using nearest-
        neighborhood matching.
        Parameters
        ---------
        weights: str or postitive definite square matrix
                 Specifies weighting matrix used in computing
                 distance meatures. 
        matches: int
                Number of matches to use for each subject.
        bias_adj: bool
                Specifies whether bias adjustments should be
                attempted.

        References
        ----------
        ..[1] Imbens, G. &Rubin, D. (2015). Causal Inference in 
                Statistics, Social, and Biomedical Sciences: An
                Introduction.
        """

        X, K = self.raw_data._dict['X'], self.raw_data._dict['K']
        X_c, X_t = self.raw_data._dict['X_c'], self.raw_data._dict['X_t']

        if weights == 'inv':
            # size = (K,)
            W = 1/X.var(0)
        elif weights == 'maha':
            V_c = np.cov(X_c, rowvar=False, ddof=0)
            V_t = np.cov(X_t, rowvar=False, ddof=0)
            if K == 1:
                # size = (1,1)
                W = 1/np.array([[(V_c+V_t)/2]])
            else:
                # size = (K, K)
                W = np.linalg.inv((V_c+V_t)/2)
        else:
            # customarize your weight
            W = weights

        self.estimates._dict['matching'] = Matching(self.raw_data, 
                                                    self.covariate_name,
                                                    W, matches, bias_adj)
        self.match_att_df = Matching(self.raw_data, self.covariate_name, W,
                                     matches, bias_adj).Get_Data(effect_type='att')
        self.match_atc_df = Matching(self.raw_data, self.covariate_name, W,
                                     matches, bias_adj).Get_Data(effect_type='atc')
        match_att = Data(self.match_att_df['PIC50'], self.match_att_df['treat'], self.match_att_df[self.covariate_name])
        match_atc = Data(self.match_atc_df['PIC50'], self.match_atc_df['treat'], self.match_atc_df[self.covariate_name])
        self.att_match_stats = Summary(match_att, self.covariate_name)
        self.atc_match_stats = Summary(match_atc, self.covariate_name)
        
    def est_propensity_s(self, lin_B=None, C_lin=1, C_qua=2.71):

        """
        Estimates the propensity score with covariates selected using
        the algorithm suggested by [1]_.

        The propensity is the conditional probability of
        receiving the treatment given the observed covariates.
        Estimation is done via a logistic regression.

        Parameters
        ----------
        lin_B: list, optional
               Column numbers(zero-based) of variables
               of the original covariate matrix X to include
               linearly. Defaults to empty list, meaning
               every column of X is subjected to the
               selection algorithm.
        C_lin: scalar, optional
               Constant value used in likelihood ratio tests
               to decide whether candidate linear terms should
               be included. Defaults to 1 as in [1]_.
        C_qua: scalar, optional
               Constant value used in likelihood ratio tests
               to decide whether candidate quadratic and interaction
               terms should be included. Defualts to 2.71 as in [1]_.

        References
        ----------
        ..[1] Imbens, G. &Rubin, D. (2015). Causal Inference in 
              Statistics, Social, and Biomedical Sciences: An 
              Introduction.
        """
        lin_basic = parse_lin_terms(self.raw_data['K'], lin_B)

        self.propensity = PropensitySelect(self.raw_data, lin_basic,
                                           C_lin, C_qua)
        self.raw_data._dict['pscore'] = self.propensity['fitted']


def parse_lin_terms(K, lin):

    if lin is None:
        return []
    elif lin == 'all':
        return range(K)
    else:
        return lin




