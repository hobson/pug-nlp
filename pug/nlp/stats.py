#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stats functions and classes useful in DOX, like `Confusion` and `cosine_distance`"""
from __future__ import print_function, division
# from past.builtins import basestring

import json
import logging
from copy import deepcopy
from itertools import product
from collections import Mapping

from scipy.optimize import minimize
import pandas as pd
from matplotlib import pyplot as plt

from pug.nlp.constant import NUMERIC_TYPES
# watch out for circular import
from pug.nlp.segmentation import stringify
from pug.nlp.util import make_dataframe, listify
from pug.nlp.util import PrettyDict
from pug.nlp.constant import INF

# from scipy import stats as scipy_stats
# import pymc

np = pd.np
logger = logging.getLogger(__name__)


def mcc_chi(mcc, num_samples):
    return np.sqrt(mcc ** 2 * num_samples) if mcc is not None and num_samples else 0
phi2chi = mcc_chi


def random_split(df, first_portion=.1):
    """Randomly sample Dataframe rows. Typically for testset/training set cross-valication

    random_split(pd.DataFrame())
    """
    mask = np.random.binomial(1, first_portion, size=len(df)).astype(bool)
    return df[mask].copy(), df[~mask].copy()


def safe_div(a, b, inf=INF):
    """Safely divide by zero and inf and nan and produce floating point result
    Numerical results are equivalent to `from __future__ import division`
    Args:
      a (float or int): numerator
      b (float or int): denominator
      inf (float or object): value to return in place of division by zero errors
      none (float or object): value to return in place of TypeErrors (division by None)
    Returns:
      dividend:
        b if b is None or NaN
        a if a is None or NaN
        inf (specified by caller, default is np.inf) if b is 0
        pd.NaT if b or a is pd.NaT
    >>> safe_div(84, 2)
    42.0
    >>> safe_div(-85, 2)
    -42.5
    >>> safe_div(42, 0)
    inf
    >>> safe_div(-42, 0)
    inf
    >>> safe_div(1, 0.)
    inf
    >>> safe_div(1, 0., inf="custom OutOfBounds str")
    'custom OutOfBounds str'
    >>> safe_div(np.inf, np.inf)
    nan
    >>> safe_div(1e200, np.inf)
    0.0
    >>> safe_div(np.inf, np.nan)
    nan
    >>> safe_div(np.nan, 0)
    inf
    >>> repr(safe_div(99.99, None))
    'None'
    >>> safe_div(42, object())
    <object object at 0x...>
    """
    try:
        return 1. * a / b
    except ZeroDivisionError:
        return inf
    except TypeError:
        try:
            1. / b
        except TypeError:
            return b
        return a
    return 1. * a / b


def mcc_chi(mcc, num_samples):
    """Return the equivalent Chi value for a given MCC (correlation) value
    >>> round(mcc_chi(0.5, 100), 3)
    5.0
    """
    return np.sqrt(mcc ** 2. * num_samples) if mcc is not None and num_samples else 0.
phi2chi = mcc_chi


def mcc_chi_squared(mcc, num_samples):
    """Return the equivalent Chi-Squared value for a given MCC (correlation) value
    Chi is a cumulative value on the horizontal axes (z-score) and represents confidence in correlation
    (like the inverse of P-value) or statistical significance.
    Assumes that the random variable being measured is continuous rather than discrete.
    Otherwise num_samples should be the number of degrees of freedom: (num_possible_states - 1) ** 2
    >>> round(mcc_chi_squared(0.5, 100), 3)
    25.0
    """
    return mcc_chi(mcc, num_samples) ** 2.


def tptnfpfn_chi(*args, **kwargs):
    """Calculate Chi from True Positive (tp), True Negative (tn), False Positive/Negative counts.
    Assumes that the random variable being measured is continuous rather than discrete.
    Reference:
      https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    >>> round(tptnfpfn_chi(1000, 2000, 30, 40))
    2765.0
    """
    tp, tn, fp, fn = args_tptnfpfn(*args, **kwargs)
    return tptnfpfn_mcc(tp=tp, tn=tn, fp=fp, fn=fn) ** 2. * (tp + tn + fp + fn)


def args_tptnfpfn(*args, **kwargs):
    """Convert kwargs for tp, tn, fp, fn to ordered tuple of args
    If a single tuple/list is passed as the first arg, it is assumed to be the desired tuple of args
    >>> args_tptnfpfn(1, 2, 3, 4)
    (1, 2, 3, 4)
    >>> args_tptnfpfn((1, 2, 3, 4))
    (1, 2, 3, 4)
    >>> args_tptnfpfn([1, 2, 3, 4])
    (1, 2, 3, 4)
    >>> args_tptnfpfn(3, 4, tp=1, tn=2)
    (1, 2, 3, 4)
    >>> args_tptnfpfn(tp=1, tn=2)
    (1, 2, 0, 0)
    >>> args_tptnfpfn(tp=1, tn=2, fp=3, fn=4)
    (1, 2, 3, 4)
    >>> args_tptnfpfn(1)
    (1, 0, 0, 0)
    """
    if len(args) == 4:
        tp, tn, fp, fn = args
    elif len(kwargs) == 0:
        if len(args) == 1:
            args = listify(args[0])
        tp, tn, fp, fn = list(list(args) + [0] * (4 - len(args)))
    else:
        args = list(args)
        tp = kwargs['tp'] if 'tp' in kwargs else args.pop(0) if len(args) else 0
        tn = kwargs['tn'] if 'tn' in kwargs else args.pop(0) if len(args) else 0
        fp = kwargs['fp'] if 'fp' in kwargs else args.pop(0) if len(args) else 0
        fn = kwargs['fn'] if 'fn' in kwargs else args.pop(0) if len(args) else 0
    return tp, tn, fp, fn


def tptnfpfn_mcc(*args, **kwargs):
    tp, tn, fp, fn = args_tptnfpfn(*args, **kwargs)
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def dataframe_tptnfpfn(df, pos_label=True, labels=None):
    """Count the True Pos, True Neg, False Pos, False Neg samples within a confusions matrx (potentiall larger than 2x2)
    >>> matrix = [[5, 3, 0], [2, 3, 1], [0, 2, 11]]
    >>> columns=['Cat', 'Dog', 'Rabbit']
    >>> x = np.array([[tc, pc] for (tc, row) in enumerate(matrix) for (pc, n) in enumerate(row) for i in range(n)])
    >>> c = Confusion([(columns[i], columns[j]) for (i, j) in x], columns=['Actual', 'Predicted'])
    >>> c
    Predicted  Cat  Dog  Rabbit
    Actual
    Cat          5    3       0
    Dog          2    3       1
    Rabbit       0    2      11
    >>> dataframe_tptnfpfn(c, 'Rabbit')
    (11, 13, 1, 2)
    >>> dataframe_tptnfpfn(c.T, 'Rabbit')
    (11, 13, 2, 1)
    >>> c.mcc[2]
    0.77901...
    """
    labels = df.columns if labels is None else labels
    neg_labels = [label for label in labels if label != pos_label]
    tp = df[pos_label][pos_label]
    tn = sum(df[pred_label][true_label] for true_label in neg_labels for pred_label in neg_labels)
    fp = df[pos_label][neg_labels].sum()
    fn = sum(df[label][pos_label] for label in neg_labels)
    return tp, tn, fp, fn


class Confusion(pd.DataFrame):
    """Compute a confusion matrix from a dataframe of true and predicted classes (2 cols)
    Stats are computed as if each of the classes were considered "positive" and
    all the others were considered "negative"
    Attributes:
      stats (dict): {
        tpr: true positive rate  = num correct positive predicitons   / num positive samples
        tnr: true negative rate  = num correct negative predictions   / num negative samples
        fpr: false positive rate = num incorrect positive predictions / num negative samples
        fnr: true positive       = num incorrect positive predictions / num positive samples
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = Positive Predictive Value = tpr / (tpr + fpr)
        npv = Negative Predictive Value= tnr / (tnr + fnr)
        plr = Pos. Likelihood Ratio = Sens. / (1 - Spec.)
        nlr = Neg. Likelihood Ratio = (1 - Sens.) / Spec. }
    Reference:
      https://en.wikipedia.org/wiki/Confusion_matrix#Example
                Predicted
             Cat Dog Rabbit
      Actual
         Cat  5   3    0
         Dog  2   3    1
      Rabbit  0   2    11
    >>> matrix = [[5, 3, 0], [2, 3, 1], [0, 2, 11]]
    >>> columns=['Cat', 'Dog', 'Rabbit']
    >>> x = np.array([[tc, pc] for (tc, row) in enumerate(matrix) for (pc, n) in enumerate(row) for i in range(n)])
    >>> c = Confusion([(columns[i], columns[j]) for (i, j) in x], columns=['Actual', 'Predicted'])
    >>> c
    Predicted  Cat  Dog  Rabbit
    Actual
    Cat          5    3       0
    Dog          2    3       1
    Rabbit       0    2      11
    >>> (Confusion(c) == c).all().all()
    True
    >>> (Confusion(x, columns=['Actual', 'Predicted']).values == matrix).all()
    True
    >>> (Confusion(x).as_data() == x).all()
    True
    >>> c.stats_dict
    {
      "tpr": 0.625,
      "fpr": 0.105263157894737,
      "tnr": 0.736842105263158,
      "fnr": 0.75,
      "plr": 5.9375,
      "nlr": 1.01785714285714,
      "accuracy": {
        "Cat": 0.18518518518518517,
        "Dog": 0.1111111111111111,
        "Rabbit": 0.40740740740740738
      },
      "sensitivity": {
        "Cat": 0.625,
        "Dog": 0.5,
        "Rabbit": 0.84615384615384615
      },
      "specificity": {
        "Cat": 0.875,
        "Dog": 0.76190476190476186,
        "Rabbit": 0.88888888888888884
      },
      "mcc": {
        "Cat": 0.54155339089324317,
        "Dog": 0.23845524161913301,
        "Rabbit": 0.77901741435486016
      },
      "chi_squared": {
        "Cat": 7.9185620300751856,
        "Dog": 1.5352443609022557,
        "Rabbit": 16.385439560439561
      }
    }

    TODO: to_data() should output 2-column DataFrame with ["Pred", "True"] columns
          constructor should allow columns and index kwargs to override category labels when values are ints
    >>> df = pd.DataFrame(zip(list('PN'*3 + 'NNNN'), list('PNNPPNNPNPPNNN')), columns=['True', 'Pred'])
    >>> c = Confusion(df, sort=False)
    >>> c.get_false_positive(scalar=False)
    P    0.333333
    N    0.428571
    dtype: float64
    >>> c.false_positive
    0.33...
    >>> c.specificity
    0.57...
    >>> c.sensitivity
    0.66...
    >>> df = pd.DataFrame(np.matrix([[1]*10,[1]*10]).T, columns=['True', 'Pred'])
    >>> c = Confusion(df)
    >>> c
    Pred   1
    True
    1     10
    >>> c.sensitivity, c.specificity
    (inf, 1.0)

    `infer=True` assumes that the first label encounted is for the **positive** class
    `infer=False` (default) assumes the only class found is the **positive** class
    So confusion matrix (and stats like senstivity) are transposed.
    >>> c = Confusion(df, infer=True)
    >>> c
    Pred  0   1
    True
    0     0   0
    1     0  10
    >>> c.sensitivity, c.specificity
    (1.0, inf)
    >>> df = pd.DataFrame(np.matrix([[0,1,2,0,1,2,1,2,2,1],[0,1,2,1,2,0,0,1,2,0]]).T, columns=['True', 'Pred'])
    >>> Confusion(df)
    Pred  0  1  2
    True
    0     1  1  0
    1     2  1  1
    2     1  1  2
    >>> str(Confusion(df.astype(str))) == str(Confusion(df))
    True
    >>> Confusion(df).sensitivity
    0    0.50
    1    0.25
    2    0.50
    dtype: float64
    >>> df = pd.DataFrame(zip(list('ABC'*5 + 'CCC'), list('ABCBCACABCACCBCCCC')), columns=['True', 'Pred'])
    >>> c = Confusion(df)
    >>> c
    Pred  A  B  C
    True
    A     1  1  3
    B     2  2  1
    C     1  1  6
    >>> c.sensitivity
    A    0.20
    B    0.40
    C    0.75
    dtype: float64
    >>> c.get_false_positive()
    A    0.80
    B    0.60
    C    0.25
    dtype: float64
    >>> c.phi == c.mcc
    A    True
    B    True
    C    True
    dtype: bool
    >>> c.mcc
    A   -0.033150
    B    0.265197
    C    0.350000
    dtype: float64
    """

    _verbose = False
    _infer = None
    _scalar_stats = None
    _sort_classes = None
    _num_classes = None
    _num_samples = None
    _colnums = None

    def __init__(self, df, *args, **kwargs):
        # 2 x 2 matrix of ints interpretted as data rather than counts/Confusion unless its a DataFrame or Confusion type
        # if ((isinstance(df, pd.DataFrame) and
        #         (getattr(df, 'columns', range(len(df))) == getattr(df, 'index', range(len(df)))).all() and
        #         len(df) == len(iter(df).next()) and
        #         all(isinstance(value), int) for row in df for value in row) and
        #         ((len(iter(df).next()) > 2) or isinstance(df, pd.DataFrame))):
        try:
            assert(isinstance(df, pd.DataFrame))
            assert((df.columns == df.index).all())
            assert(df.values.dtype == int)
            self.construct_copy(df, *args, **kwargs)
            return
        except (AssertionError, AttributeError, ValueError):
            pass
        _infer = kwargs.pop('infer', kwargs.pop('infer_classes', None))
        _sort_classes = kwargs.pop('sort', kwargs.pop('sort_classes', 1))
        verbose = kwargs.pop('verbose', False)

        df = make_dataframe(df, *args, **kwargs)
        # only 2 columns allowed: true class and predicted class
        columns = df.columns[:2]
        # to maintain the order of classes:
        # pd.Index(df[df.columns[0]]).intersection(pd.Index(df[df.columns[1]])).unique
        # or pip install orderedset by Raymond Hettinger and Rob Speer
        index = pd.Index(np.concatenate([df[columns[0]], df[columns[1]]])).unique()

        if _infer and len(index) == 1:
            index = np.concatenate([index, [infer_pos_label(index[0])]])
            # if verbose:
            #     print('WARN: Only a single class label was found} so inferring (guessing) a positive label.')
        index = pd.Index(pd.Index(index).unique())
        if _sort_classes:
            index = index.sort_values(ascending=(_sort_classes > 0))

        # construct an empty parent DataFrame instance
        super(Confusion, self).__init__(index=pd.Index(index, name=columns[0]), columns=pd.Index(index, name=columns[1]))

        # metadata to speed other operations
        self._verbose = verbose
        self._infer = _infer
        self._scalar_stats = kwargs.pop('scalar', kwargs.pop('scalar_stats', None))
        self._sort_classes = _sort_classes
        self._num_classes = len(index)
        self._num_samples = len(df)
        self._colnums = np.arange(0, self._num_classes)
        # look for Positive and Negative column labels by first finding columns labeled
        #    "Negative", "-1", "0", "Healthy", "N/A", etc

        try:
            self._neg_label = (label for label in self.columns if unicode(label).strip().lower()[0] in ('-nh0')).next()
        except StopIteration:
            self._neg_label = self.columns[-1]
        try:
            self._pos_label = (label for label in self.columns if label != self._neg_label).next()
        except StopIteration:
            self._pos_label = infer_pos_label(self._neg_label)

        logger.debug('true class samples: {}'.format(df[columns[0]].values[:5]))
        for p_class in index:
            self[p_class] = pd.Series([len(df[(df[columns[0]] == t_class) & (df[columns[1]] == p_class)]) for t_class in index],
                                      index=index, dtype=int)
        self.refresh_meta()

    def construct_copy(self, other, *args, **kwargs):
        # construct a parent DataFrame instance
        parent_type = super(Confusion, self)
        parent_type.__init__(other)
        try:
            for k, v in other.__dict__.iteritems():
                if hasattr(parent_type, k) and hasattr(self, k) and getattr(parent_type, k) == getattr(self, k):
                    continue
                setattr(self, k, deepcopy(v))
        except AttributeError:
            pass
        self.refresh_meta()

    def as_data(self):
        return np.array([[tc, pc] for (tc, row) in enumerate(self.values) for (pc, n) in enumerate(row) for i in range(n)])

    def refresh_meta(self):
        """Calculations that only depend on aggregate counts in Confusion Matrix go here"""

        # these calcs are duplicated in __init__()
        self._num_classes = len(self.index)
        self._colnums = np.arange(0, self._num_classes)
        try:
            self._neg_label = (label for label in self.columns if unicode(label).strip().lower()[0] in ('-nh0')).next()
        except StopIteration:
            self._neg_label = self.columns[-1]
        try:
            self._pos_label = (label for label in self.columns if label != self._neg_label).next()
        except StopIteration:
            self._pos_label = infer_pos_label(self._neg_label)

        # TODO: reorder columns with newly guessed pos and neg class labels first

        # TODO: gather up additional meta calculations here so
        #       a Confusion matrix can be build from an existing DataFrame that contains confusion counts
        #       rather than just two columns of labels.
        self._hist_labels = self.sum().astype(int)
        self._num_total = self._hist_labels.sum()
        assert(self._num_total == self.sum().sum())
        self._num_pos_labels = self._hist_labels.get(self._pos_label, 0)
        self._num_neg_labels = self._num_total - self._num_pos_labels  # everything that isn't positive is negative
        self._hist_classes = self.T.sum()
        self._num_pos = self._hist_classes.get(self._pos_label, 0)
        self._num_neg = self._hist_classes.sum() - self._num_pos  # everything that isn't positive is negative
        self._tp = self.get(self._pos_label, pd.Series()).get(self._pos_label, 0)
        self._tpr = safe_div(float(self._tp), self._num_pos)
        self._tn = np.diag(self).sum() - self._tp
        self._tnr = safe_div(float(self._tn), self._num_neg)
        self._fp = self.get(self._pos_label, pd.Series()).sum() - self._tp
        self._fpr = safe_div(float(self._fp), self._num_neg)
        self._fn = self._num_neg_labels - self._tn
        self._fnr = safe_div(float(self._fn), self._num_pos)
        self._plr = safe_div(float(self._tpr), self._fpr)
        self._nlr = safe_div(float(self._fnr), self._tnr)
        self._binary_accuracy = safe_div(self._tp + self._tn, self._num_samples)
        self._binary_sensitivity = safe_div(self._tp, self._tp + self._fn)
        self._binary_specificity = safe_div(self._tn, self._tn + self._fp)
        # # These asserts intentionally commented out. They are just developer FYI.
        # # May fail due to platform-dependent floating point rounding implementations
        # assert self._binary_sensitivity == self._tpr
        # assert self._binary_specificity == self._tnr

    @classmethod
    def from_existing(cls, confusion, *args, **kwargs):
        """Creates a confusion matrix from a DataFrame that already contains confusion counts (but not meta stats)
        >>> df = pd.DataFrame(np.matrix([[0,1,2,0,1,2,1,2,2,1],[0,1,2,1,2,0,0,1,2,0]]).T, columns=['True', 'Pred'])
        >>> c = Confusion(df)
        >>> c2 = pd.DataFrame(c)
        >>> hasattr(c2, '_binary_sensitivity')
        False
        >>> c3 = Confusion.from_existing(c2)
        >>> hasattr(c3, '_binary_sensitivity')
        True
        >>> (c3 == c).all().all()
        True
        >>> c3
        Pred  0  1  2
        True
        0     1  1  0
        1     2  1  1
        2     1  1  2
        """
        # Extremely brute-force to recreate data from a confusion matrix!

        df = []
        for t, p in product(confusion.index.values, confusion.columns.values):
            df += [[t, p]] * confusion[p][t]
        if confusion.index.name is not None and confusion.columns.name is not None:
            return Confusion(pd.DataFrame(df, columns=[confusion.index.name, confusion.columns.name]))
        return Confusion(pd.DataFrame(df))

    def get_accuracy(self, scalar=None):
        """Num_True_Positive / Num_Samples"""
        # to get a Series instead of a dict:
        # (np.diag(c).astype(float) / c.T.sum())
        #     == pd.Series(self.sensitivity)
        if ((not self._scalar_stats and not scalar and self._num_classes > 2) or
                ((scalar is False or self._scalar_stats is False) and self._num_classes > 1)):
            return pd.Series(PrettyDict([(k, safe_div(self[k][k], self._num_total)) for k in self.columns]))
        return self._binary_accuracy
    accuracy = property(get_accuracy)

    def get_sensitivity(self, scalar=None):
        """True Positive Rate = TP / P = Num_True_Positive / (Num_True_Postive + Num_False_Negative)"""
        # to get a Series instead of a dict:
        # (np.diag(c).astype(float) / c.T.sum())
        #     == pd.Series(self.sensitivity)
        if ((not self._scalar_stats and not scalar and self._num_classes > 2) or
                ((scalar is False or self._scalar_stats is False) and self._num_classes > 1)):
            return pd.Series(PrettyDict([(k, safe_div(self[k][k], self.loc[k].sum())) for k in self.columns]))
        return self._binary_sensitivity
    sensitivity = property(get_sensitivity)

    def get_specificity(self, scalar=None):
        """True_Negative / (True_Negative + False_Positive)"""
        if ((not self._scalar_stats and not scalar and self._num_classes > 2) or
                ((scalar is False or self._scalar_stats is False) and self._num_classes > 1)):
            spec = PrettyDict()
            for pos_label in self.columns:
                neg_labels = [label for label in self.columns if label != pos_label]
                tn = sum(self[label][label] for label in neg_labels)
                # fp = self[pos_label][neg_labels].sum()
                fp = self.loc[neg_labels].sum()[pos_label]
                assert(self[pos_label][neg_labels].sum() == fp)
                spec[pos_label] = float(tn) / (tn + fp)
            return pd.Series(spec)
        return self._binary_specificity
    specificity = property(get_specificity)

    def get_phi(self, scalar=None):
        """Phi (φ) Coefficient -- lack of confusion
        Arguments:
          scalar (bool or None): Whether to return a scalar Phi coefficient (assume binary classification) rather than a multiclass vector
        Measure of the lack of confusion in a single value
        References:
          [MCC on wikipedia](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
          [docs on R implementation](http://www.personality-project.org/r/html/phi.html)
        φ =   (TP*TN - FP*FN) / sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
        mcc = (tp*tn - fp*fn) / sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
        """
        # If requested, compute the phi coeffients for all possible 'positive' and 'negative' class labels (multiclass problem)
        if ((not self._scalar_stats and not scalar and self._num_classes > 2) or
                ((scalar is False or self._scalar_stats is False) and self._num_classes > 1)):
            phi = PrettyDict()
            # count of predictions labeled with pred_label for a slice of data that was actually labeled true_label:
            # `count = self[pred_label][true_label]`
            for pos_label in self.columns:
                tp, tn, fp, fn = dataframe_tptnfpfn(self, pos_label=pos_label, labels=self.columns)
                phi[pos_label] = tptnfpfn_mcc(tp=tp, tn=tn, fp=fp, fn=fn)
            return pd.Series(phi)
        # A scalar phi value was requested, so compute it for the "inferred" positive classification class
        return tptnfpfn_mcc(self._tp, self._tn, self._fp, self._fn)
    phi = property(get_phi)
    # mcc = Matthews Correlation Coefficient = phi coefficient
    get_mcc = get_phi
    mcc = property(get_mcc)

    def get_chi(self, scalar=None):
        """sqrt(Chi_Squared) statistic (see `mcc`, `phi`, or google 'Matthews Correlation Coefficient'"""
        phi = self.get_phi(scalar=scalar)
        return mcc_chi(phi, self._num_samples)
    chi = property(get_chi)

    def get_chi_squared(self, scalar=None):
        return self.get_chi(scalar=scalar) ** 2
    chi_squared = property(get_chi_squared)

    def get_binary_confusion(self, pos_label=None):
        pos_label = pos_label if pos_label in self.columns else self._pos_label
        neg_labels = [label for label in self.columns if label != pos_label]
        columns = [pos_label, '+'.join(str(lbl) for lbl in neg_labels)]
        conf = pd.DataFrame(index=columns, columns=columns)
        return conf

    def get_false_positive(self, scalar=True):
        """Normalized false positive rate (0 <= fp <= 1)"""
        ans = pd.Series(PrettyDict([(k, safe_div(np.sum(self.loc[k][[j for j in self.columns if j != k]]), np.sum(self.loc[k])))
                                   for k in self.columns]))
        if (not self._scalar_stats and not scalar) or self._num_classes != 2:
            return ans
        return ans[self._pos_label]
    false_positive = property(get_false_positive)
    fpr = false_positive

    def get_false_negative(self, scalar=True):
        """Normalized false positive rate (0 <= fp <= 1)"""
        ans = pd.Series(PrettyDict([(k, safe_div(np.sum(self.loc[k]) - self[k][k], np.sum(self.sum() - self.loc[k]))) for k in self.columns]))
        if (not self._scalar_stats and not scalar) or self._num_classes != 2:
            return ans
        return ans[self._pos_label]
    false_negative = property(get_false_negative)
    fnr = false_negative

    def get_stats_dict(self, scalar=False):
        # TPR and TNR, etc should be vectors so each set of stats can be a column in a DataFrame
        # TPDO: make this a PrettyDict around a list comprehension over the attribute names
        d = PrettyDict([
                       ('tpr',         self._tpr if isinstance(self._tpr, NUMERIC_TYPES) else PrettyDict(self._tpr)),
                       ('fpr',         self._fpr if isinstance(self._fpr, NUMERIC_TYPES) else PrettyDict(self._fpr)),
                       ('tnr',         self._tnr if isinstance(self._tnr, NUMERIC_TYPES) else PrettyDict(self._tnr)),
                       ('fnr',         self._fnr if isinstance(self._fnr, NUMERIC_TYPES) else PrettyDict(self._fnr)),
                       ('plr',         self._plr if isinstance(self._plr, NUMERIC_TYPES) else PrettyDict(self._plr)),
                       ('nlr',         self._nlr if isinstance(self._nlr, NUMERIC_TYPES) else PrettyDict(self._nlr)),
                       ('accuracy',    self.accuracy if isinstance(self.accuracy, NUMERIC_TYPES) else
                        PrettyDict([(label, self.accuracy[label]) for label in self.columns])),
                       ('sensitivity', self.sensitivity if isinstance(self.sensitivity, NUMERIC_TYPES) else
                        PrettyDict([(label, self.sensitivity[label]) for label in self.columns])),
                       ('specificity', self.specificity if isinstance(self.specificity, NUMERIC_TYPES) else
                        PrettyDict([(label, self.specificity[label]) for label in self.columns])),
                       ('mcc',         self.mcc if isinstance(self.mcc, NUMERIC_TYPES) else
                        PrettyDict([(label, self.mcc[label]) for label in self.columns])),
                       ('chi_squared', self.chi_squared if isinstance(self.chi_squared, NUMERIC_TYPES) else
                        PrettyDict([(label, self.chi_squared[label]) for label in self.columns])),
                       ])
        if not scalar:
            return d
        return PrettyDict([(k, v[self._pos_label]) for k, v in d.iteritems()])
    stats_dict = property(get_stats_dict)

    def get_stats(self):
        df = pd.DataFrame()
        for k, v in self.get_stats_dict(scalar=False).iteritems():
            if isinstance(v, Mapping):
                df[k] = pd.Series(v)
        return df

    def to_markdown(self):
        markdown = ''
        markdown += '## Confusion Matrices with "{}" as Truth\n\n'.format(self.columns.name)
        markdown += '### {}-vs-{}\n\n'.format(self.index.name, self.columns.name)
        markdown += self.to_html() + '\n'
        markdown += self.get_stats(scalar=False).to_html() + '\n\n'
        return markdown

    def __repr__(self):
        s = super(Confusion, self).__repr__()
        return(s)


# TODO: reuse inverse dict function and fuzzy_get from pug.nlp
POS_LABELS = {'0': '1', 'False': 'True', 'F': 'T', 'No': 'Yes', 'N': 'P',
              'None': 'Positive', 'Neg': 'Pos', 'Negative': 'Positive', "A": "B"}
POS_LABELS_INVERSE = dict((v, k) for k, v in POS_LABELS.iteritems())
POS_LABELS_LOWER = dict((k.lower(), v.lower()) for k, v in POS_LABELS.iteritems())
POS_LABELS_LOWER_INVERSE = dict((v.lower(), k.lower()) for k, v in POS_LABELS.iteritems())
POS_LABELS_LOWER_INVERSE = dict((k.lower(), v.lower()) for k, v in POS_LABELS.iteritems())
POS_LABELS_LOWER_FIRST = dict((k.lower()[0], v.lower()[0]) for k, v in POS_LABELS.iteritems())
POS_LABELS_LOWER_INVERSE_FIRST = dict((v.lower()[0], k.lower()[0]) for k, v in POS_LABELS.iteritems())
# POS_LABELS_ALL = OrderedDict(list(POS_LABELS.iteritems()) + list(POS_LABELS_LOWER.iteritems())


def infer_pos_label(neg_label=None):
    """Try to guess a positive classification label from a negative label
    Basis for an NLP function to find the "opposite" of a string
    yes->no, true->false, etc)
    >>> [infer_pos_label(x) for x in ('1', '-1', '0', '2', 1, 0, 'F', 'False', False, True)]
    ['0', '1', '1', '3', 0, 1, 'T', 'True', True, False]
    """
    # A class label should be a None, bool, int, or str
    if neg_label is None:
        return True
    typ = type(neg_label)
    # If class label isn't a bool or None then make it an int or str
    try:
        neg_label = int(float(neg_label))
        if neg_label in (0, 1):
            return typ(int(not neg_label))
        if neg_label < 0:
            return typ(-neg_label)
        return typ(neg_label + 1)
    except:
        neg_label = stringify(neg_label).strip()
    for xform, label_dict in zip((lambda x: x, lambda x: x,       str.lower,        str.lower,),
                                 (POS_LABELS, POS_LABELS_INVERSE, POS_LABELS_LOWER, POS_LABELS_LOWER_INVERSE)):
        try:
            return typ(label_dict[xform(neg_label)])
        except KeyError:
            pass
    # neg_label = neg.lower()
    # for labels in (POS_LABELS_LOWER, POS_LABELS_LOWER_INVERSE):
    #     try:
    #         return typ(labels[neg_label])
    #     except KeyError:
    #         pass
    # if not neg_label:
    #     return True
    # neg = neg[0]
    # try:
    #     return POS_LABELS_LOWER_FIRST[neg]
    # except KeyError:
    #     pass
    return 'P'


CLASSES = [str(i) for i in range(11)]
CLASSES = dict(zip(CLASSES, range(len(CLASSES))))


# TODO: Create a dict or Pandas Index of node names and their d3 group index (0-N)
def graph_from_cov_df(df, threshold=.5, gain=2., n=None, class_dict=CLASSES):
    """Compose pair of lists of dicts (nodes, edges) for the graph described by a DataFrame"""
    n = n or len(df)
    nodes = [{'group': class_dict.get(name, 0), "name": name} for name in df.index.values][:n]
    edges = []
    for i, (row_name, row) in enumerate(df.iterrows()):
        for j, value in enumerate(row.values):
            if i > j and value * gain > threshold and i < n and j < n:
                edges += [{'source': i, 'target': j, 'value': gain * value}]
    return nodes, edges


def json_from_cov_df(df, threshold=.5, gain=2., n=None, indent=1):
    """Produce a json string describing the graph (list of edges) from a square auto-correlation/covariance matrix
       { "nodes": [{"group": 1, "name": "the"},
                {"group": 1, "name": "and"},
                {"group": 1, "name": "our"},
                {"group": 2, "name": "that"},...
         "links": [{"source": 0, "target": 0, "value": 2.637520131294177},
                   {"source": 0, "target": 1, "value": 1.343999676850537}, ...
    """
    nodes, edges = graph_from_cov_df(df=df, threshold=threshold, gain=gain, n=n)
    return json.dumps({'nodes': nodes, 'links': edges}, indent=indent)


# # TODO: only do this when STRICT or DEBUG or VERBOSE global variables are set (in django.settings?)
# # raise exceptions for numpy and pandas warnings:
# warnings.filterwarnings('error')


def spec_from_thresh(thresh, labels, scores, *args, **kwargs):
    r"""Compute the specifity that a particular threshold on the scores can acheive
    specificity = Num_True_Negative / (Num_True_Negative + Num_False_Positive)
    >>> scores = np.arange(0, 1, 0.1)
    >>> spec_from_thresh(0.5, labels=(scores > .5).astype(int), scores=scores)
    1.0
    >>> spec_from_thresh(0.5, labels=(scores > .4).astype(int), scores=scores)
    1.0
    >>> spec_from_thresh(0.5, labels=(scores > .9).astype(int), scores=scores)
    0.6
    """
    df = pd.DataFrame(zip(labels, np.array(scores > thresh).astype(int)))
    c = Confusion(df, *args, **kwargs)
    return c._binary_specificity


def sens_from_thresh(thresh, labels, scores, *args, **kwargs):
    r"""Compute the specifity that a particular threshold on the scores can acheive
    sensitivity = Num_True_Positive / (Num_True_Postive + Num_False_Negative)
    >>> scores = np.arange(0, 1, 0.1)
    >>> sens_from_thresh(0.5, labels=(scores > .5).astype(int), scores=scores)
    1.0
    >>> sens_from_thresh(0.5, labels=(scores > .4).astype(int), scores=scores)
    0.8
    """
    df = pd.DataFrame(zip(labels, np.array(scores > thresh).astype(int)))
    c = Confusion(df, *args, **kwargs)
    return c._binary_sensitivity


def sens_from_spec(spec, labels, scores, *args, **kwargs):
    r"""Find the sensitivity that corresponds to the indicated specificity (ROC function)
    sensitivity = Num_True_Positive / (Num_True_Postive + Num_False_Negative)
    specificity = Num_True_Negative / (Num_True_Negative + Num_False_Positive)
    """
    thresh = thresh_from_spec(spec, labels, scores)
    df = pd.DataFrame(zip(labels, scores[scores > thresh.astype(int)]))
    c = Confusion(df, *args, **kwargs)
    return c._binary_sensitivity


def cost_fun(x, *args, **kwargs):
    """Enables minimization (root-finding) for non-zero targets
    Returns regularized absolute error between target and output of `fun`.
    Configurable state with function attributes.
    >>> cost_fun.fun = spec_from_thresh  # "forward" function we want to evaluate in cost_fun to compute error
    >>> cost_fun.target = 1.0  # desired specificity
    >>> thresh, labels, scores = 0.9, [0]*10, np.arange(0, 1, 0.1)
    >>> spec = spec_from_thresh(thresh, labels, scores)
    >>> spec  # should get perfect specificity because no samples were wrongly labeled
    1.0
    >>> cost_fun(thresh, labels, scores)  # should be zero cost, since target = function output
    0.0
    >>> spec = spec_from_thresh(.5, labels, scores)
    >>> spec  # should get less than perfect specificity now that there are some incorrect neg predictions
    0.6
    >>> 1.0 - spec  # so the "cost" of this incorrect threshold should be the error in the specificity
    0.4
    >>> round(cost_fun(spec, labels, scores), 6)  # should be approximately same as error in spec above
    0.4
    """
    cost_fun.regularization_weight = kwargs.pop('regularization_weight', cost_fun.regularization_weight)
    cost_fun.verbose = verbose = kwargs.pop('verbose', cost_fun.verbose)
    fun = cost_fun.fun = kwargs.get('fun', None) or cost_fun.fun
    target = cost_fun.target = cost_fun.target if kwargs.get('target', None) is None else kwargs.pop('target')
    delta = target - fun(x, *args)
    cost = np.abs(delta) + getattr(cost_fun, 'regularization_weight', 0) * np.abs(x)
    if verbose:
        print("x, target = {}, {} => delta = {} - {} => cost = abs({}) + abs({}) = {}".format(
            x, target, target, fun(x, *args), delta, x, cost))
    return cost
cost_fun.fun = spec_from_thresh
cost_fun.target = 0
cost_fun.verbose = False
cost_fun.regularization_weight = 0


def plot_cost(scores=np.random.rand(100), thresh=0.5, noise=0):
    """Plot the cost function topology (contours for each of several targets)"""
    c = pd.DataFrame(index=np.arange(0, 1, 0.01))
    if isinstance(thresh, (int, float)):
        thresh = [thresh]
    elif not isinstance(thresh, (pd.Series, np.ndarray, list, tuple)):
        thresh = np.arange(0, 1, .2)
    cost_fun.fun = spec_from_thresh
    for t in thresh:
        labels = (scores / t / scores.max() / 1.00001).astype(int)
        cost_fun.target = t
        c['target=0.{}'.format(int(t * 10))] = np.array([cost_fun(x, labels, scores, verbose=True) for x in c.index])
    c.plot()
    plt.show(block=False)


def fun_inverse(fun=None, y=0, x0=None, args=(), disp=False, method='Nelder-Mead', **kwargs):
    r"""Find the threshold level that accomplishes the desired specificity
    Call indicated function repeatedly to find answer to the inverse function evaluation
    Arguments:
      fun (function): function to be calculate an inverse for
      y (float): desired output of fun
      x0 (float): initial guess at input to fun, the fun arg that will be adjusted
      args (list or tuple): constant arguments to fun which will not be adjusted
      constraints (tuple): dictionary of optimizer constraints (see scipy.optimize.minimize)
      disp (bool): whether to display incremental results during optimization
      method (str): one of the scipy.optimize.minimize methods
      additional kwargs are passed along to the minimize function
    fun_inverse can be used to calculate a trivial square root:
    >>> round(fun_inverse(fun=lambda x: x**2, y=9, x0=0), 6)
    3.0
    """
    fun_inverse.fun = cost_fun.fun = fun if fun is not None else getattr(fun_inverse, 'fun', lambda x: x)
    fun_inverse.target = cost_fun.target = y or 0
    fun_inverse.verbose = verbose = cost_fun.verbose = kwargs.pop(
        'verbose', getattr(cost_fun, 'verbose', getattr(fun_inverse, 'verbose', False)))
    fun_inverse.x0 = x0 = x0 if x0 is not None else getattr(fun_inverse, 'x0', 0) or 0

    if verbose:
        print('    x0: {}\ntarget: {}\n'.format(fun_inverse.x0, fun_inverse.target))

    res = minimize(cost_fun,
                   x0=x0,
                   args=args,
                   options=kwargs.pop('options', {}),
                   method=method,
                   **kwargs
                   )
    if isinstance(x0, NUMERIC_TYPES):
        return res.x[0]
    return res.x


def thresh_from_spec(spec, labels, scores, **kwargs):
    r"""Find the threshold level that accomplishes the desired specificity"""
    cost_fun.verbose = kwargs.pop('verbose', cost_fun.verbose)
    cost_fun.target = spec
    return minimize(cost_fun,
                    x0=[.5],
                    args=(labels, scores),
                    method='SLSQP',
                    constraints=({'type': 'ineq',
                                  'fun': lambda x: np.array([x[0]]),
                                  'jac': lambda x: np.array([1.])},
                                 ),
                    **kwargs
                    )


def write_graph_json(filename='similarity_graph.json', verbosity=1):
    js = json_from_cov_df(df=make_dataframe())
    if verbosity:
        print(js[:1000] + '\n...\n' + js[-1000:])
    with open(filename, 'w') as f:
        f.write(js)


def dyno_hist(x, window=None, probability=True, edge_weight=1.):
    """ Probability Distribution function from values
    Arguments:
      probability (bool): whether the values should be min/max scaled to lie on the range [0, 1]
    Like `hist` but smoother, more accurate/useful
    Double-Normalization:
      The x values are min/max normalized to lie in the range 0-1 inclusive
      The pdf is normalized to integrate/sum to 1.0
    >>> h = dyno_hist(np.arange(100), window=5)
    >>> abs(sum(np.diff(h.index.values) * h.values[1:]) - 1.) < 0.00001
    True
    >>> h = dyno_hist(np.arange(50), window=12)
    >>> abs(sum(np.diff(h.index.values) * h.values[1:]) - 1.) < 0.00001
    True
    >>> h = dyno_hist(np.random.randn(1000), window=42)
    >>> abs(sum(np.diff(h.index.values) * h.values[1:]) - 1.0) < 0.04
    True
    """
    x = np.sort(x)
    if probability:
        # normalize x first
        x = x - x[0]
        x = x / float(x[-1] or 1)
    window = window or 1
    window = min(max(window, 1), int(len(x) / 1.5))
    window += 1
    # Empirical Densitites (PDF) based on diff of sorted values
    delta = x[(window - 1):] - x[:(1 - window)]
    densities = float(window - 1) / (len(delta) + window - 2) / delta
    h = pd.Series(densities, index=x[window // 2:][:len(delta)])
    if probability:
        if h.index[0] > 0:
            h = pd.Series(edge_weight * densities[0], index=[0]).append(h)
        if h.index[-1] < 1:
            h = h.append(pd.Series(edge_weight * densities[-1], index=[1.]))
    return h
