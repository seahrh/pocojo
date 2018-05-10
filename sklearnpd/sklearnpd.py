from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import numpy as np


class TransformPipeline(Pipeline):
    def get_feature_names(self):
        for step in self.steps[::-1]:
            t = step[1]
            if hasattr(t, 'get_feature_names'):
                # print(f'step={repr(step)}')
                return t.get_feature_names()
        raise ValueError(
            'At least one transformer in the pipeline must implement `get_feature_names` method')


class TransformLatentDirichletAllocation(LatentDirichletAllocation):
    def get_feature_names(self):
        return list(range(1, self.n_components + 1))


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
       https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
       Also see Zac Stewart's excellent blogpost on pipelines:
       http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
       """

    def __init__(self, col, as_type, as_matrix=False):
        self.col = col
        self.as_type = as_type
        self.as_matrix = as_matrix

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        # set the array type to be given type
        res = np.asarray(df[self.col]).astype(self.as_type)
        if self.as_matrix is True:
            res = res.reshape(-1, 1)
        print(f'{repr(self.col)} ColumnExtractor shape={repr(np.shape(res))}')
        return res

    def fit(self, *_):
        return self

    def get_feature_names(self):
        return [self.col]


class PrefixColumnExtractor(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
       https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
       Also see Zac Stewart's excellent blogpost on pipelines:
       http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
       """

    def __init__(self, prefix, as_type):
        self.prefix = prefix
        self.filter_col = None
        self.as_type = as_type

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        self.filter_col = [col for col in df if col.startswith(self.prefix)]
        res = np.asarray(df[self.filter_col]).astype(self.as_type)
        print(f'{repr(self.prefix)} PrefixColumnExtractor shape={repr(np.shape(res))}')
        return res

    def fit(self, *_):
        return self

    def get_feature_names(self):
        return self.filter_col


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
