from collections import defaultdict
import numpy as np
from pandas import DataFrame, Series

class NaiveBayes(object):
    """
    Represents naive bayes classifier. Instances are constructed with training
    data and options to configure the classifier.

    Parameters
    ----------
    training_data: sequence of tuples as [(label, [features])]
      e.g., [("good", ["hello", "world"])]
    log: boolean, default False
      Whether to transform the computed probabilities into log space
    smoothing: float, default 0.
      Amount of additive smoothing
    """
    def __init__(self, training_data, log=False, smoothing=0.):
        """
        Parameters
        ----------
        training_data: sequence of tuples as [(label, [features])]
          e.g., [("good", ["hello", "world"])]
        log: boolean, default False
          Whether to transform the computed probabilities into log space
        smoothing: float, default 0.
          Amount of additive smoothing
        """
        self.log = log
        self.smoothing = smoothing
        self.train(training_data)
        self.cond_dist = self.train(training_data)
        
    def train(self, data):
        """
        Re-train the classifier with given input training data.
        The effects of previous training are overridden

        Parameters
        ----------
        data: sequence of tuples as [(label, [features])]
          e.g., [("positive", ["hello", "world"])]
          
        """
        self._curr_data = data
        if data is not None:
            self._label_dist, self._cond_dist = train_data(data,
                                                           self.smoothing)
            self._labels = self._label_dist.samples
        return self._cond_dist

    def classify(self, data, prob=False):
        """
        Classify a given set of features, optionally return the estimated
        probability

        Parameters
        ----------
        data: sequence of str [features]
          e.g., ["hello", "world"]
        prob: boolean, default False
          Whether to return the estimated probability as well
        """
        if data is None or len(data) == 0:
            if prob:
                return None, np.nan
            else:
                return None

        post = Series(self._calc_post(data))
        max_label = post.idxmax()
        if not prob:
            return max_label
        return max_label, post[max_label]

    def _calc_post(self, data):
        """
        Parameters
        ----------
        data: list or ndarray
        prob: bool, default True
          Also return probabilities if true
        """
        data = self._discard_missing(data)
        label_priors = self._calc_label_priors()
        post_raw = self._handle_conditional(label_priors, data)
        post = self._normalize(post_raw)
        return post

    def _discard_missing(self, data):
        clean_data = []
        for feature in data:
            for label in self._labels:
                if (label, feature) in self._cond_dist:
                    clean_data.append(feature)
                    break

        return clean_data

    @property
    def _prob_fname(self):
        return 'logprob' if self.log else 'prob'

    @property
    def agg(self):
        import operator as op
        return op.add if self.log else op.mul

    @property
    def _label_pfunc(self):
        return getattr(self._label_dist, self._prob_fname)

    def _get_cond_pfunc(self, label, feature):
        pdist = self._cond_dist[(label, feature)]
        return getattr(pdist, self._prob_fname)

    # separate function in case we need to tweak this
    def _calc_label_priors(self):
        return self._label_pfunc()

    def _handle_conditional(self, priors, data):
        post = priors.copy()
        [self._label_cond_helper(label, data, post) for label in self._labels]
        return post

    def _label_cond_helper(self, label, data, prob):
        # ! SIDE EFFECT
        for feature in data:
            value = True # BoW
            pfunc = self._get_cond_pfunc(label, feature)
            prob[label] = self.agg(prob[label], pfunc(value))

    def _normalize(self, post):
        if self.log:
            value_sum = np.log(post).sum()
            if not np.isfinite(value_sum):
                post[:] = np.log(1.0 / len(post))
            else:
                post -= value_sum
        else:
            value_sum = post.sum()
            if value_sum == 0:
                post[:] = 1.0 / len(post)
            else:
                post /= value_sum
        return post

class ProbDist(object):

    def __init__(self, value_counts, smooth=0.):
        self._value_counts = Series(value_counts)
        self._N = self._value_counts.sum()
        self._bins = len(self._value_counts)
        #self._smooth = 0.5
        self._smooth = smooth
    @property
    def total(self):
        return self._N + self._bins * self._smooth

    @property
    def samples(self):
        return self._value_counts.index

    def prob(self, feature=None):
        """
        Parameters
        ----------
        feature: object, optional
          If None or unspecified then returns prob for all distinct samples
        """
        p = (self._value_counts.get(feature, 0)
             if feature else self._value_counts)
        return (p + self._smooth) / self.total

    def logprob(self, feature=None):
        """
        Parameters
        ----------
        feature: object, optional
          If None or unspecified then returns log prob for all distinct samples
        """
        p = (self._value_counts.get(feature, 0)
             if feature else self._value_counts)
        return (np.log(p + self._smooth) - np.log(self.total))


def train_data(data, smoothing=0.):
    """
    Parameters
    ----------
    data: nested dict
    """
    (label_counts, feature_values,
     conditional, conditional_count) = _gather_freq_counts(data)

    # side-effect!
    _fill_missing_features(label_counts, feature_values,
                           conditional, conditional_count)

    label_prior = ProbDist(label_counts, smoothing)
    cond_dist = {(label, feature) : ProbDist(fcounts, smoothing)
                 for ((label, feature), fcounts)
                 in conditional.iteritems()}

    return label_prior, cond_dist

def _gather_freq_counts(data):
    """
    computes:
    1. num occurrences by label
    2. set of distinct values by feature
    3. num samples per value by label/feature
    4. num samples per label/feature
    """
    label_counts = defaultdict(int)
    feature_values = defaultdict(set)
    conditional = defaultdict(lambda: defaultdict(int))
    conditional_count = defaultdict(int)

    for label, featureset in data:
        label_counts[label] += 1
        for feature in featureset:
            value = True # BoW
            conditional[(label, feature)][value] += 1
            conditional_count[(label, feature)] += 1
            feature_values[feature].add(value)

    return (label_counts, feature_values,
            conditional, conditional_count)

def _fill_missing_features(label_counts, feature_values,
                           conditional, conditional_count):
    """
    Compute features that do NOT occur for a label

    Notes
    -----
    Has side effect
    """
    for label, nsamples in label_counts.iteritems():
        for feature in feature_values.iterkeys():
            count = conditional_count[(label, feature)]
            conditional[(label, feature)][False] = nsamples - count
            feature_values[feature].add(False)
