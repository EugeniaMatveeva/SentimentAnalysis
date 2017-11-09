import random

from nltk.corpus import movie_reviews
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import *
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import pandas as pd

from evaluation import *
from parameter_tuning import evaluate
from transformations.nltk_preprocessor import NLTKPreprocessor

now = datetime.datetime.now()

def data_load(shuffle=True):
    X = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
    y = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]
    Xy = zip(X, y)
    if shuffle:
        random.shuffle(Xy, )
    else:
        random.shuffle(Xy, lambda: 0.42 )
    return [x[0] for x in Xy], [x[1] for x in Xy]

def compare_classifiers(vectorizer, data, target, vectorizer_name=None):
    vectorizer_name = vectorizer_name or type(vectorizer).__name__
    pipe_logreg= Pipeline([(vectorizer_name, vectorizer),
                           ('logisticregression', LogisticRegression())])
    log_pipeline_results(pipe_logreg, evaluate(pipe_logreg, data, target))

    pipe_svc = Pipeline([(vectorizer_name, vectorizer),
                         ('linearsvc', LinearSVC())])
    log_pipeline_results(pipe_svc, evaluate(pipe_svc, data, target))

    pipe_sgd = Pipeline([(vectorizer_name, vectorizer),
                         ('sgdclassifier, loss=hinge, penalty=l2, alpha=1e-3', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
    log_pipeline_results(pipe_sgd, evaluate(pipe_sgd, data, target))

    pipe_mlp = Pipeline([(vectorizer_name, vectorizer),
                         ('mlp, (30,30,30)', MLPClassifier(hidden_layer_sizes=(30,30,30)))])
    log_pipeline_results(pipe_mlp, evaluate(pipe_mlp, data, target))

    pipe_nb = Pipeline([(vectorizer_name, vectorizer),
                        ('multinomialnb', MultinomialNB())])
    log_pipeline_results(pipe_nb, evaluate(pipe_nb, data, target))

    pipe_rf = Pipeline([(vectorizer_name, vectorizer),
                        ('randomforest', RandomForestClassifier(n_jobs=-1, oob_score=True))])
    log_pipeline_results(pipe_rf, evaluate(pipe_rf, data, target))


X, y = data_load()

cnt_vectorizer = CountVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), min_df=2)
tf_vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), min_df=2)

evaluator = Evaluator(preprocessor=NLTKPreprocessor(add_PosNeg=True), vectorizer=tf_vectorizer, load_path='12grams')
# params_grid = {'classifier:mlp__hidden_layer_sizes': [(30), (100)],
#                'classifier:mlp__activation': ('logistic', 'tanh', 'relu'),
#                'classifier:mlp__solver': ['lbfgs'],
#                'classifier:mlp__alpha': [0.01, 0.001, 0.0001]
#                }
# evaluator.choose_classifier_parameters(MLPClassifier(), 'mlp', X, y, params_grid)
evaluator.test_classifier(MLPClassifier(hidden_layer_sizes=(30), activation='logistic', solver='lbfgs', alpha=0.001), 'mlp', X, y, test_split=0)



