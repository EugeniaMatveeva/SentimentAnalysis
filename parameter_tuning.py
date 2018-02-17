from validation import *
from helper import *
from transformations.nltk_preprocessor import NLTKPreprocessor
from transformations.inquirerlex_transform import InquirerLexTransform
from vader_analyzer import SentimentAnalyzer

import numpy as np
from sklearn.pipeline import make_union
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

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
        score = validate(pp, X, y, cv=n_cv).mean()
        scores[pp.steps[1][0]] = score
        print 'Pipeline %s: accuracy mean = %f' % (pp.named_steps.keys(), score)
    return scores


def evaluate(pipe, X, y, cv=n_cv):
    return validate(pipe, X, y, cv).mean()


def estimate_stop_words(stop_words, classifier, X, y):
    scores = {}
    vectorizer = CountVectorizer()
    for lbl, stop in stop_words.items():
        vectorizer.stop_words=stop
        pipeline = Pipeline([('countvectorizer', vectorizer),
                             (type(classifier).__name__, classifier)])
        scores[lbl] = validate(pipeline, X, y, cv=n_cv).mean()
        print 'Score for %s: %f' % (lbl, scores[lbl])
    return scores


def estimate_n_grams(classifier, X, y, n):
    vectorizer = CountVectorizer(ngram_range=[1, n])
    vectorizer.fit_transform(X, y)
    pipeline = Pipeline([('countvectorizer', vectorizer),
                         (type(classifier).__name__, classifier)])
    score_n_grams = validate(pipeline, X, y, cv=n_cv).mean()
    print 'Score for %i grams: %f' % (n, score_n_grams)
    return score_n_grams


def estimate_n_lettergrams(classifier, X, y, n_min, n_max):
    vectorizer = CountVectorizer(ngram_range=[n_min, n_max], analyzer='char_wb')
    vectorizer.fit_transform(X, y)
    pipeline = Pipeline([('countvectorizer', vectorizer),
                         (type(classifier).__name__, classifier)])
    score_n_grams = validate(pipeline, X, y, cv=n_cv).mean()
    print 'Score for letter %i to %i grams: %f' % (n_min, n_max, score_n_grams)
    return score_n_grams

def test_pipelines_params1(X, y):
    # sentiment simple:
    # 'Best params: ', {'cnt__ngram_range': (1, 2), min_df=0, max_df=0.75, max_features=10000, penalty='elasticnet', loss='log', n_iter=80}) Best score: 0.761500
    # sentiment full:
    # 'Best params: ', {'cnt__ngram_range': (1, 3), 'cnt__max_features': None, 'cnt__min_df': 0, 'clf__alpha': 1e-06, 'clf__n_iter': 80,
    # 'clf__penalty': 'elasticnet', 'clf__loss': 'log', 'cnt__max_df': 0.9}) Best score: 0.855162
    pipe = Pipeline([
        ('cnt', CountVectorizer(min_df=0, max_df=0.9, max_features=None, ngram_range=(1, 3))),
        ('clf', SGDClassifier(penalty='elasticnet', loss='log', n_iter=80, alpha=1e-06)),
    ])
    params_grid = {#'cnt__ngram_range': ((1,1), (1,2), (1,3)),
                   #'cnt__min_df': (0, 1, 5),
                   #'cnt__max_df': (0.75, 0.9, 1),
                   #'cnt__max_features': (10000, None),
                   #'clf__alpha': (0.00001, 0.000001, 1e-6),
                   #'clf__penalty': ('l1','l2', 'elasticnet'),
                   #'clf__loss': ('hinge','log', 'perceptron'),
                   #'clf__n_iter': (50, 80),
                   }
    res = grid_search(pipe, params_grid, X, y)
    print ('Best params: ', res.best_params_)
    print 'Best score: %f' % res.best_score_
    return pipe

def test_pipelines_params2(X, y):
    # sentiment simple:
    # 'Best params: ', {'tfidf__max_features': None}) Best score: 0.780000
    # sentiment full:
    # 'Best params: ', {'tfidf__stop_words': 'english', 'clf__loss': 'hinge', 'tfidf__ngram_range': (1, 2), 'tfidf__max_df': 0.75, 'clf__penalty': 'l2',
    # 'clf__alpha': 1e-05, 'clf__n_iter': 50}) Best score: 0.867101/0.919361
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=0, max_df=0.9, norm='l2',ngram_range=(1,2), stop_words=None, max_features=None)),
        ('clf', SGDClassifier(penalty='l2', loss='hinge', alpha=1e-5, n_iter=50)),
    ])
    params_grid = {#'tfidf__ngram_range': ((1,2), (1,3)),
                   #'tfidf__min_df': (0, 1),
                   #'tfidf__max_df': (0.75, 0.9, 1),
                   #'tfidf__max_features': (1000, 5000, 10000, None),
                   #'tfidf__stop_words' : ('english', None),
                   #'tfidf__norm': ('l1', 'l2', None),
                   #'clf__alpha': (0.00001, 0.000001, 1e-6),
                   #'clf__penalty': ('l1','l2', 'elasticnet'),
                   #'clf__loss': ('hinge','log', 'perceptron'),
                   #'clf__n_iter': (10, 50, 80),
                   }
    res = grid_search(pipe, params_grid, X, y)
    print ('Best params: ', res.best_params_)
    print 'Best score: %f' % res.best_score_
    return pipe


def test_pipelines_params3(X, y):
    # sentiment simple:
    # 'Best params: ', {'clf__tol': 0.9, 'clf__C': 150, 'clf__max_iter': 20, 'clf__solver': 'liblinear'}) Best score: 0.783500
    # sentiment full:
    # 'Best params: ', {'tfidf__stop_words': 'english', 'tfidf__min_df': 0, 'tfidf__max_features': None, 'clf__tol': 1, 'clf__C': 150, 'clf__solver': 'newton-cg'}) Best score: 0.868066
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=0, max_df=0.75, norm='l2',ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(solver='newton-cg', C=150, tol=0.9)),
    ])
    params_grid = {#'tfidf__ngram_range': ((1,2), (1,3)),
                   #'tfidf__min_df': (0, 2, 10),
                   #'tfidf__max_df': (0.75, 0.9, 1),
                   #'tfidf__max_features': (5000, 10000, None),
                   #'tfidf__stop_words' : ('english', None),
                   #'tfidf__norm': ('l1', 'l2', None),
                   #'clf__C': (100, 150, 180),
                   #'clf__solver': ('newton-cg', 'liblinear', 'sag'),
                   #'clf__penalty': ('l2'),
                   #'clf__tol': (1, 0.9, 0.75),
                   #'clf__max_iter': (10, 50, 80),
                   }
    res = grid_search(pipe, params_grid, X, y)
    print ('Best params: ', res.best_params_)
    print 'Best score: %f' % res.best_score_
    return pipe

    
def test_pipelines_params4(X, y):
    # sentiment simple:
    # 'Best params: ', {solver='sgd', activation='tanh', learning_rate_init=0.5, learning_rate='constant' 'clf__hidden_layer_sizes': 100, 'clf__learning_rate_init': 0.1}) Best score: 0.783000, Best score: 0.779500
    # sentiment full:
    # 'Best params: ', {'tfidf__max_df': 0.9, 'tfidf__ngram_range': (1, 2), 'tfidf__stop_words': None, 'tfidf__max_features': None, 'tfidf__norm': 'l2', 'clf__solver': 'sgd',
    # 'clf__momentum': 0.8, 'clf__learning_rate': 'constant'}) Best score: 0.869513
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=0, max_df=0.9, norm='l2',ngram_range=(1,2), stop_words=None, max_features=None)),
        ('clf', MLPClassifier(hidden_layer_sizes=(30), solver='sgd', activation='tanh', learning_rate_init=0.1, learning_rate='constant', momentum=0.8)),

    ])
    params_grid = {#'tfidf__ngram_range': ((1,2), (1,3)),
                   #'tfidf__min_df': (0, 2, 10),
                   #'tfidf__max_df': (0.75, 0.9, 1),
                   #'tfidf__max_features': (5000, 10000, None),
                   #'tfidf__stop_words' : ('english', None),
                   #'tfidf__norm': ('l1', 'l2', None),
                   #'clf__activation': ('logistic', 'tanh', 'relu'),
                   #'clf__solver': ('lbfgs', 'sgd', 'adam'),
                   #'clf__alpha': (0.01, 0.001, 0.0001),
                   #'clf__learning_rate': ('constant', 'adaptive'),
                   #'clf__learning_rate_init': (0.5, 0.1, 0.01),
                   #'clf__max_iter': (100, 200, 500),
                   #'clf__momentum': (0.8, 0.9, 1)
                   #'clf__hidden_layer_sizes': ((30), (100), (70, 30))
                   }
    res = grid_search(pipe, params_grid, X, y)
    print ('Best params: ', res.best_params_)
    print 'Best score: %f' % res.best_score_
    return pipe

def test_pipelines_params5(X, y):
    # sentiment simple:
    # 'Best params: ', {'clf__tol': 0.001, 'clf__C': 100, 'clf__max_iter': 10000}) Best score: 0.778500
    # sentiment full:
    #('Best params: ', {'tfidf__ngram_range': (1, 2), 'tfidf__min_df': 0, 'tfidf__stop_words': None, 'tfidf__max_df': 0.75, 'tfidf__norm': 'l2', 'clf__tol': 0.1, 'clf__C': 100}) Best score: 0.868548
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=0, max_df=0.75, norm='l2',ngram_range=(1,2), stop_words=None)),
        ('clf', LinearSVC(penalty='l2', loss='squared_hinge', tol=0.1, C=100, max_iter=10000)),
    ])
    params_grid = {#'tfidf__ngram_range': ((1,2), (1,3)),
                   #'tfidf__min_df': (0, 1, 2, 10),
                   #'tfidf__max_df': (0.75, 0.9, 1),
                   # 'tfidf__max_features': (1000, 5000, 10000, None),
                   #'tfidf__stop_words' : ('english', None),
                   #'tfidf__norm': ('l1', 'l2', None),
                   #'clf__tol': (0.1, 0.01, 1e-3),
                   #'clf__C': (20, 50, 100),
                   #'clf__max_iter': (1000, 10000),
                   }
    res = grid_search(pipe, params_grid, X, y)
    print ('Best params: ', res.best_params_)
    print 'Best score: %f' % res.best_score_
    return pipe

def test_pipelines_params6(X, y):
    # sentiment simple:
    # 'Best params: ', {'clf__fit_prior': False, 'clf__alpha': 0.5}) Best score: 0.784000
    # sentiment full:
    # 'Best params: ', {'cnt__max_features': None, 'cnt__max_df': 0.75, 'cnt__stop_words': None, 'cnt__ngram_range': (1, 3), 'clf__fit_prior': True, 'clf__alpha': 0.75}) Best score: 0.865412\
    pipe = Pipeline([
        ('cnt', CountVectorizer(min_df=0, max_df=0.75, ngram_range=(1,3), stop_words=None, max_features=None)),
        ('clf', MultinomialNB(fit_prior=True, alpha=0.75)),
    ])
    params_grid = {#'cnt__ngram_range': ((1,2), (1,3)),
                   #'cnt__min_df': (0, 1, 2, 10),
                   #'cnt__max_df': (0.75, 0.9, 1),
                   #'cnt__max_features': (1000, 5000, 10000, None),
                   #'cnt__stop_words' : ('english', None),
                   #'clf__alpha': (0, 0.25, 0.5, 0.75, 1),
                   #'clf__fit_prior': (True, False),
                   }
    res = grid_search(pipe, params_grid, X, y)
    print ('Best params: ', res.best_params_)
    print 'Best score: %f' % res.best_score_
    return pipe

def test_pipelines_params7(X, y):
    # sentiment simple:
    # 'Best params: ', {'clf__hidden_layer_sizes': 100, 'clf__activation': 'tanh', 'clf__solver': 'sgd', 'clf__momentum': 0.9, 'clf__learning_rate': 'adaptive', 'clf__alpha': 0.001}) Best score: 0.768500
    # sentiment full:
    #('Best params: ', {'clf__activation': 'logistic', 'clf__solver': 'sgd', 'clf__momentum': 0.9, 'clf__learning_rate_init': 0.01, 'clf__alpha': 0.0001,
    # 'clf__learning_rate': 'constant', 'clf__hidden_layer_sizes': 100, 'clf__activation': 'logistic'}) Best score: 0.855282
    pipe = Pipeline([
        ('cnt', CountVectorizer(min_df=0, max_df=0.75, max_features=10000, ngram_range=(1,2))),
        ('clf', MLPClassifier(hidden_layer_sizes=(100), solver='sgd', activation='logistic', learning_rate_init=0.01, momentum=0.9, alpha=0.0001, learning_rate='constant')),
    ])
    params_grid = {#'cnt__ngram_range': ((1,1), (1,2), (1,3)),
                   #'cnt__min_df': (0, 1, 5),
                   #'cnt__max_df': (0.75, 0.9, 1),
                   #'cnt__max_features': (10000, None),
                   #'clf__activation': ('logistic', 'tanh', 'relu'),
                   #'clf__solver': ('lbfgs', 'sgd', 'adam'),
                   #'clf__alpha': (0.01, 0.001, 1e-4),
                   #'clf__learning_rate': ('constant', 'adaptive'),
                   #'clf__learning_rate_init': (0.5, 0.1, 0.01),
                   #'clf__max_iter': (100, 200, 500),
                   #'clf__momentum': (0.8, 0.9, 1),
                   #'clf__hidden_layer_sizes': ((30), (100), (70, 30))
                   }
    res = grid_search(pipe, params_grid, X, y)
    print ('Best params: ', res.best_params_)
    print 'Best score: %f' % res.best_score_
    return pipe

def get_complex_pipeline(classifier, pipeline_type=None, preprocessor = NLTKPreprocessor(), vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), min_df=2)):
    if pipeline_type == 'use_lex':
        lex_vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)
        pipeline2 = Pipeline([('inq', InquirerLexTransform()), ('inq_tf', lex_vectorizer)])
        ext = [vectorizer, pipeline2]
        vectorizer = make_union(*ext)
        model = Pipeline([
            ('nltk', preprocessor),
            ('tf_lex', vectorizer),
            ('norm', Normalizer()),
            ('clf', classifier),
        ])
    elif pipeline_type == 'use_nltk_vader':
        sent_vectorizer = SentimentAnalyzer()
        pipeline1 = Pipeline([('nltk', preprocessor), ('tf', vectorizer)])
        ext = [pipeline1, sent_vectorizer]
        vectorizer = make_union(*ext)
        model = Pipeline([
            ('tf_vader', vectorizer),
            ('clf', classifier),
        ])
    elif pipeline_type == 'use_vader':
        sent_vectorizer = SentimentAnalyzer()
        pipeline1 = Pipeline([('tf', vectorizer)])
        ext = [pipeline1, sent_vectorizer]
        vectorizer = make_union(*ext)
        model = Pipeline([
            ('tf_vader', vectorizer),
            ('clf', classifier),
        ])
    elif pipeline_type == 'only_vader':
        sent_vectorizer = SentimentAnalyzer()
        model = Pipeline([
            ('vader', sent_vectorizer),
            ('clf', classifier),
        ])
    else:
        model = Pipeline([
            ('nltk', preprocessor),
            ('tf', vectorizer),
            ('clf', classifier),
        ])
    return model


def test_complex_pipeline_params(X, y):
    # 'use_lex' - Best score: 0.856488
    # 'use_nltk_vader' - ('Best params: ', {'tf_vader__pipeline__nltk__filter_token': True}) Best score: 0.862880
    # 'use_vader'- ('Best params: ', {'tf_vader__pipeline__tf__ngram_range': (1, 3), 'tf_vader__pipeline__tf__norm': 'l2', 'tf_vader__pipeline__tf__stop_words': 'english'}) Best score: 0.856126
    #               'tf_vader__pipeline__tf__ngram_range': (1, 2), 'tf_vader__pipeline__tf__norm': 'l2', 'tf_vader__pipeline__tf__stop_words': None}) Best score: 0.867101
    # 'nltk'- ('Best params: ', {'tf__ngram_range': (1, 2), 'tf__stop_words': None, 'tf__norm': 'l2'}) Best score: 0.868186
    #         {'nltk__filter_token': False, 'clf__activation': 'tanh','clf__learning_rate_init': 0.5, 'clf__learning_rate': 'adaptive', 'clf__alpha': 0.01}) Best score: 0.872648

    clf = MLPClassifier(solver='sgd', activation='tanh', learning_rate_init=0.5, learning_rate='adaptive', momentum=0.9, alpha=0.01)
    #prep = NLTKPreprocessor(add_PosNeg=False, replace_SynSet=False, filter_token=False)
    tfidf = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), min_df=0, max_df=0.75, stop_words=None, max_features=None, norm='l2')
    pipe = get_complex_pipeline(clf, 'nltk', vectorizer=tfidf)
    params_grid = {
                   #'nltk__filter_token': (True, False),
                   #'tf_vader__pipeline__tf__ngram_range': ((1,2), (1,3)),
                   #'tf_vader__pipeline__tf__min_df': (0, 2, 10),
                   #'tf_vader__pipeline__tf__max_df': (0.75, 0.9, 1),
                   #'tf_vader__pipeline__tf__max_features': (1000, 5000, 10000, None),
                   #'tf_vader__pipeline__tf__stop_words' : ('english', None),
                   #'tf_vader__pipeline__tf__norm': ('l1', 'l2', None),
                   #'clf__activation': ('logistic', 'tanh', 'relu'),
                   #'clf__solver': ('lbfgs', 'sgd', 'adam'),
                   #'clf__alpha': (0.01, 0.001, 0.0001),
                   #'clf__learning_rate': ('constant', 'adaptive'),
                   #'clf__learning_rate_init': (0.5, 0.1),
                   #'clf__max_iter': (100, 200, 500),
                   #'clf__momentum': (0.8, 0.9, 1)
                   }
    res = grid_search(pipe, params_grid, X, y)
    print ('Best params: ', res.best_params_)
    print 'Best score: %f' % res.best_score_
    return pipe