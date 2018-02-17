import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from evaluation import *
from loading import *
from parameter_tuning import evaluate
from transformations.nltk_preprocessor import NLTKPreprocessor
import parameter_tuning as ptune

def compare_classifiers(vectorizer, X, y, vectorizer_name=None):
    vectorizer_name = vectorizer_name or type(vectorizer).__name__
    pipe_logreg= Pipeline([(vectorizer_name, vectorizer),
                           ('logisticregression', LogisticRegression())])
    log_pipeline_results(pipe_logreg, evaluate(pipe_logreg, X, y))

    pipe_svc = Pipeline([(vectorizer_name, vectorizer),
                         ('linearsvc', LinearSVC())])
    log_pipeline_results(pipe_svc, evaluate(pipe_svc, X, y))

    pipe_sgd = Pipeline([(vectorizer_name, vectorizer),
                         ('sgdclassifier, loss=hinge, penalty=l2, alpha=1e-3', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
    log_pipeline_results(pipe_sgd, evaluate(pipe_sgd, X, y))

    pipe_mlp = Pipeline([(vectorizer_name, vectorizer),
                         ('mlp, (30,30,30)', MLPClassifier(hidden_layer_sizes=(30,30,30)))])
    log_pipeline_results(pipe_mlp, evaluate(pipe_mlp, X, y))

    pipe_nb = Pipeline([(vectorizer_name, vectorizer),
                        ('multinomialnb', MultinomialNB())])
    log_pipeline_results(pipe_nb, evaluate(pipe_nb, X, y))

    pipe_rf = Pipeline([(vectorizer_name, vectorizer),
                        ('randomforest', RandomForestClassifier(n_jobs=-1, oob_score=True))])
    log_pipeline_results(pipe_rf, evaluate(pipe_rf, X, y))


def compare_pipelines(X, y):
    ptune.test_pipelines_params1(X, y)
    ptune.test_pipelines_params2(X, y)
    ptune.test_pipelines_params3(X, y)
    ptune.test_pipelines_params4(X, y)
    ptune.test_pipelines_params5(X, y)
    ptune.test_pipelines_params6(X, y)
    ptune.test_pipelines_params7(X, y)
    ptune.test_complex_pipeline_params(X, y)


def evaluate_movie_reviews():
    X, y = load_moview_reviews(True)
    tf_vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), min_df=2)

    evaluator = Evaluator(preprocessor=NLTKPreprocessor(add_PosNeg=True), vectorizer=tf_vectorizer, load_path='12grams')
    params_grid = {'classifier:mlp__hidden_layer_sizes': [(30), (100)],
                   'classifier:mlp__activation': ('logistic', 'tanh', 'relu'),
                   'classifier:mlp__solver': ['lbfgs'],
                   'classifier:mlp__alpha': [0.01, 0.001, 0.0001]
                   }
    evaluator.choose_model_parameters(MLPClassifier(), 'mlp', X, y, params_grid)
    evaluator.test_classifier(MLPClassifier(hidden_layer_sizes=(30), activation='logistic', solver='lbfgs', alpha=0.001), 'mlp', X, y, test_split=0)


def evaluate_product_reviews():
    X_train, y_train = load_product_reviews('data/', False)
    df_test = load_product_reviews_test('data/')
    X_test = df_test.ix[:]['text'].values
    log_classifier = LogisticRegression(n_jobs=-1)
    # params_grid = {'classifier:log__penalty': ('l1', 'l2'),
    #                'classifier:mlp__activation': ('logistic', 'tanh', 'relu'),
    #                'classifier:mlp__solver': ['lbfgs'],
    #                'classifier:mlp__alpha': [0.01, 0.001, 0.0001]
    #                }
    sgd_classifier = SGDClassifier(penalty='l2', loss='hinge', alpha=1e-5, n_iter=80)
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(30), activation='logistic', solver='sgd')

    clf = MLPClassifier(solver='sgd', activation='tanh', learning_rate_init=0.5, learning_rate='constant', momentum=0.9, alpha=0.01)
    tfidf = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,3), min_df=0, max_df=0.75, stop_words=None, max_features=None)
    evaluator = Evaluator(vectorizer=tfidf, model_type='use_vader')
    evaluator.test_classifier(sgd_classifier, 'sgd', X_train, y_train)
    y_test = evaluator.get_predictions(sgd_classifier, X_train, y_train, X_test)
    df_test['y'] = y_test
    write_predictions('data/product_sentiment_prediction.csv', df_test)

######

X_train, y_train = load_reviews('data/reviews_train.csv', balance=True)
df_test = load_reviews_xml('data/reviews_test.xml')
X_test = df_test.ix[:]['text'].values

pipeline = ptune.test_complex_pipeline_params(X_train, y_train)
pipeline.fit(X_train, y_train)
y_test = pipeline.predict(X_test)
df_test['y'] = ['neg' if y == 0 else 'pos' for y in y_test]
write_predictions('data/phone_reviews_sentiment_prediction_nltk_unbalanced.csv', df_test)

