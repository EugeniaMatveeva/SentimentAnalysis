from sklearn.model_selection import cross_val_score


def validate(clf, X, y, cv=3, scoring=None):
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    return scores
