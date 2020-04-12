# -*- coding: utf-8 -*-

'''
    author: Zhewei Song
    Created on 03/05/2020
'''

import numpy as np 
import Xcausal.utils.tools as tools


class Summary(object):

    """
    Dictionay-like class containing summary statistics for input data.

    One of the summary statistics is the normal difference between
    covariates. Large values indicate that simple linear adjustment methods
    may not be adequate for removing biases that are associated with 
    differences in covariates.
    """

    def __init__(self, data, covariate_name):

        self.covariate_name = covariate_name
        self._dict = dict()

        self._dict['N'], self._dict['K'] = data._dict['N'], data._dict['K']
        self._dict['N_c'], self._dict['N_t'] = data._dict['N_c'], data._dict['N_t']
        self._dict['Y_c_mean'] = data._dict['Y_c'].mean()
        self._dict['Y_t_mean'] = data._dict['Y_t'].mean()
        self._dict['Y_c_sd'] = np.sqrt(data._dict['Y_c'].var(ddof=1))
        self._dict['Y_t_sd'] = np.sqrt(data._dict['Y_t'].var(ddof=1))
        self._dict['rdiff'] = self._dict['Y_t_mean'] - self._dict['Y_c_mean']
        self._dict['X_c_mean'] = data._dict['X_c'].mean(0)
        self._dict['X_t_mean'] = data._dict['X_t'].mean(0)
        self._dict['X_c_sd'] = np.sqrt(data._dict['X_c'].var(0, ddof=1))
        self._dict['X_t_sd'] = np.sqrt(data._dict['X_t'].var(0, ddof=1))
        self._dict['ndiff'] = calc_ndiff(self._dict['X_c_mean'],
                                         self._dict['X_t_mean'],
                                         self._dict['X_c_sd'],
                                         self._dict['X_t_sd'])

    def __str__(self):

        table_width = 80

        N_c, N_t, K = self._dict['N_c'], self._dict['N_t'], self._dict['K']
        Y_c_mean, Y_t_mean = self._dict['Y_c_mean'], self._dict['Y_t_mean']
        Y_c_sd, Y_t_sd = self._dict['Y_c_sd'], self._dict['Y_t_sd']
        X_c_mean, X_t_mean = self._dict['X_c_mean'], self._dict['X_t_mean']
        X_c_sd, X_t_sd = self._dict['X_c_sd'], self._dict['X_t_sd']
        rdiff, ndiff = self._dict['rdiff'], self._dict['ndiff']
        varnames = self.covariate_name

        output = '\n'
        output += 'Summary Statistics\n\n'

        entries1 = ['', 'Controls (N_c='+str(N_c)+')',
                    'Treated (N_t='+str(N_t)+')', '']
        entry_types1 = ['string']*4
        col_spans1 = [1, 2, 2, 1]
        output += tools.add_row(entries1, entry_types1,
                                col_spans1, table_width)

        entries2 = ['Variable', 'Mean', 'S.d.', 
                    'Mean', 'S.d.', 'Raw-diff']
        entry_types2 = ['string']*6
        col_spans2 = [1]*6
        output += tools.add_row(entries2, entry_types2,
                                col_spans2, table_width)
        output += tools.add_line(table_width)

        entries3 = ['Y', Y_c_mean, Y_c_sd, Y_t_mean, Y_t_sd, rdiff]
        entry_types3 = ['string'] + ['float']*5
        col_spans3 = [1]*6
        output += tools.add_row(entries3, entry_types3,
                                col_spans3, table_width)

        output += '\n'
        output += tools.add_row(entries1, entry_types1,
                                col_spans1, table_width)

        entries4 = ['Variable', 'Mean', 'S.d.',
                    'Mean', 'S.d.', 'Nor-diff']
        output += tools.add_row(entries4, entry_types2, 
                                col_spans2, table_width)
        output += tools.add_line(table_width)

        entry_types5 = ['string'] + ['float']*5
        col_spans5 = [1]*6
        for entries5 in zip(varnames, X_c_mean, X_c_sd,
                            X_t_mean, X_t_sd, ndiff):
            output += tools.add_row(entries5, entry_types5,
                                    col_spans5, table_width)

        return output
            





def calc_ndiff(mean_c, mean_t, sd_c, sd_t):

    return (mean_t-mean_c) / np.sqrt((sd_c**2 + sd_t**2)/2)