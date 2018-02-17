from sklearn.base import  TransformerMixin

class Transformer(TransformerMixin):
    ''' Base class for pure transformers that don't need a fit method '''
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return X

    def get_params(self, deep=True):
        return dict()