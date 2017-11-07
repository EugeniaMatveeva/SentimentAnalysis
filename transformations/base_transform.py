class StatelessTransform:
    """
    Base class for all transformations that do not depend on training (ie, are
    stateless).
    """
    def fit(self, X, y=None):
        return self