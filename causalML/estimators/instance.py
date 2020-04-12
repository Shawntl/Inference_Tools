# -*- coding: utf-8 -*-

'''
    author: Zhewei Song
    Created on 03/07/2020
'''

from Xcausal.utils import tools


class OutFormat(object):

    """
    Dictionary-like class containing treatemnt effect estimates.
    """
    def __str__(self):
        table_width = 80

        names = ['ate', 'atc', 'att']
        coefs = [self._dict[name] for name in names if name in self._dict.keys()]
        ses = [self._dict[name+'_se'] for name in names if name+'_se' in self._dict.keys()]
        output = '\n'
        output += 'Treatment Effect Estimates: ' + self._method + '\n\n'

        entries1 = ['', 'Est.', 'S.e.', 'z', 'P>|z|', 
                    '[95% Conf. int.]']
        entry_types1 = ['string']*6
        col_spans1 = [1]*5 + [2]
        output += tools.add_row(entries1, entry_types1,
                                col_spans1, table_width)
        output += tools.add_line(table_width)

        entry_types2 = ['string'] + ['float']*6
        col_spans2 = [1]*7
        for (name, coef, se) in zip(names, coefs, ses):
            entries2 = tools.gen_reg_entries(name.upper(), coef, se)
            output += tools.add_row(entries2, entry_types2,
                                    col_spans2, table_width)

        return output


class Estimators(object):

    """
    Dictionary-like class containing treatment effect estimates.
    """

    def __init__(self):

        self._dict = {}

    def __setitem__(self, key, item):

        self._dict[key] = item

    def __str__(self):

        output = ''
        for method in self._dict.keys():
            output += self._dict[method].__str__()

        return output




