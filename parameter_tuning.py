import helper
from validation import *

import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

n_cv = 5


def compare_accuracy(pipes, X, y):
    scores = []
    for pp in pipes:
        score = np.array(cross_val_score(pp, X, y, cv=n_cv))
        scores.append(score)
        print 'Pipeline %s: accuracy mean = %f' % (pp.named_steps.keys(), score.mean())
        print 'Pipeline %s: accuracy std = %f' % (pp.named_steps.keys(), score.std())
    return scores


def choose_countvectorizer_min_df(clf_pipe, values, X, y):
    params_grid = {'countvectorizer__min_df': values}
    res = grid_search(clf_pipe, params_grid, X, y)
    print 'Best min_df value: %f' % res.best_params_['countvectorizer__min_df']
    print 'Best score: %f' % res.best_score_
    return res


def grid_search(clf, params_grid, X, y):
    gs = GridSearchCV(clf, params_grid)
    gs.fit(X, y)
    return gs


def choose_classifier(pipes, X, y):
    scores = {}
    for pp in pipes:
        scores[pp.steps[1][0]] = validate(pp, X, y, cv=n_cv).mean()
    return scores


def evaluate(pipe, X, y, cv=n_cv):
    return validate(pipe, X, y, cv).mean()


def estimate_stop_words(stop_words, classifier, X, y):
    scores = {}
    vectorizer = CountVectorizer(stop_words=stop_words[0])
    for lbl, stop in stop_words.items():
        vectorizer.stop_words=stop_words
        pipeline = Pipeline([('countvectorizer', vectorizer),
                                ('classifier', classifier)])
        scores[lbl] = validate(pipeline, X, y, cv=n_cv).mean()
        print 'Score for %s: %f' % (lbl, scores[lbl])
    return scores


def estimate_n_grams(classifier, X, y, n):
    vectorizer = CountVectorizer(ngram_range=[1, n])
    vectorizer.fit_transform(X, y)
    pipeline = Pipeline([('countvectorizer', vectorizer),
                         ('logreg', classifier)])
    score_n_grams = validate(pipeline, X, y, cv=n_cv).mean()
    print 'Score for %i grams: %f' % (n, score_n_grams)
    return score_n_grams


def estimate_n_lettergrams(classifier, X, y, n_min, n_max):
    vectorizer = CountVectorizer(ngram_range=[n_min, n_max], analyzer='char_wb')
    vectorizer.fit_transform(X, y)
    pipeline = Pipeline([('countvectorizer', vectorizer),
                         ('logreg', classifier)])
    score_n_grams = validate(pipeline, X, y, cv=n_cv).mean()
    print 'Score for letter %i to %i grams: %f' % (n_min, n_max, score_n_grams)
    return score_n_grams