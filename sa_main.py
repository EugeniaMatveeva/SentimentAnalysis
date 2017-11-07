import random

from nltk.corpus import movie_reviews
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import *
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from evaluation import *
from parameter_tuning import evaluate
from transformations.nltk_preprocessor import NLTKPreprocessor

now = datetime.datetime.now()


def data_load(shuffle=True):
    X = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
    y = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]
    Xy = zip(X, y)
    if shuffle:
        random.seed(42)
        random.shuffle(Xy, )
    return [x[0] for x in Xy], [x[1] for x in Xy]

def compare_classifiers(vectorizer, data, target, vectorizer_name=None):
    vectorizer_name = vectorizer_name or type(vectorizer).__name__
    pipe_cnt_logreg = Pipeline([(vectorizer_name, vectorizer),
                                ('logisticregression', LogisticRegression())])
    log_pipeline_results(pipe_cnt_logreg, evaluate(pipe_cnt_logreg, data, target))

    pipe_cnt_svc = Pipeline([(vectorizer_name, vectorizer),
                              ('linearsvc', LinearSVC())])
    log_pipeline_results(pipe_cnt_svc, evaluate(pipe_cnt_svc, data, target))

    pipe_cnt_sgd = Pipeline([(vectorizer_name, vectorizer),
                             ('sgdclassifier, loss=hinge, penalty=l2, alpha=1e-3', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
    log_pipeline_results(pipe_cnt_sgd, evaluate(pipe_cnt_sgd, data, target))

    pipe_cnt_mlp = Pipeline([(vectorizer_name, vectorizer),
                             ('mlp, (30,30,30)', MLPClassifier(hidden_layer_sizes=(30,30,30)))])
    log_pipeline_results(pipe_cnt_mlp, evaluate(pipe_cnt_mlp, data, target))

    pipe_cnt_nb = Pipeline([(vectorizer_name, vectorizer),
                            ('multinomialnb', MultinomialNB())])
    log_pipeline_results(pipe_cnt_nb, evaluate(pipe_cnt_nb, data, target))


X, y = data_load()
# Label encode the targets
labels = LabelEncoder()
y = labels.fit_transform(y)

cnt_vectorizer = CountVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), min_df=2)
tf_vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,1), min_df=2)
sgd_classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3)
rfc = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)

save_paths = {
    'preprocessor': 'preprocessor',
    'vectorizer': 'vectorizer'
}
load_paths = {
    'preprocessor': 'preprocessor',
    'vectorizer': 'vectorizer'
}


evaluator = Evaluator(preprocessor=NLTKPreprocessor(add_PosNeg=True), vectorizer=tf_vectorizer, use_inqlex=False, use_test_split=False)
#test([X[0]], [y[0]])
evaluator.test_classifier(sgd_classifier, 'sgd_classifier', X, y, show_errors=False)
# evaluator.test_classifier(rfc, 'RandomForestClassifier', X, y, show_errors=False)
# evaluator.test_classifier(MLPClassifier(hidden_layer_sizes=100), 'mlp, hidden_layers=(100)', X, y)



# documents = zip(data, target)
# nltk_classifier = NltkClassifier(documents)
# nltk_classifier.test_classifier()

# cnt_vectorizer = CountVectorizer(min_df=2)
# cnt_vectorizer.fit_transform(X)
# tf_vectorizer = TfidfVectorizer()

# compare_classifiers(cnt_vectorizer, X, y, 'countvectorizer, min_df=2')
# compare_classifiers(tf_vectorizer, X, y, 'tfidfvectorizer')




