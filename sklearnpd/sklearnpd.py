from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np


class TransformPipeline(Pipeline):
    def get_feature_names(self):
        last = self.steps[-1]
        print(f'last={repr(last)}')
        return last[1].get_feature_names()


class TextExtractor(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
       https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
       Also see Zac Stewart's excellent blogpost on pipelines:
       http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
       """

    def __init__(self, col):
        self.col = col

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        # set the array type to be string
        return np.asarray(df[self.col]).astype(str)

    def fit(self, *_):
        return self


class PrefixColumnExtractor(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
       https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
       Also see Zac Stewart's excellent blogpost on pipelines:
       http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
       """

    def __init__(self, prefix):
        self.prefix = prefix

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        filter_col = [col for col in df if col.startswith(self.prefix)]
        return np.asarray(df[filter_col])

    def fit(self, *_):
        return self


class Apply(BaseEstimator, TransformerMixin):
    """Applies a function f element-wise to the numpy array
    """

    def __init__(self, fn):
        self.fn = fn

    def transform(self, data):
        # note: reshaping is necessary because otherwise sklearn
        # interprets 1-d array as a single sample
        fnv = np.vectorize(self.fn)
        return fnv(data.reshape(data.size, 1))

    def fit(self, *_):
        return self
