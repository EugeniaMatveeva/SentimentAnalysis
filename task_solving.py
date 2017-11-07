import helper
from parameter_tuning import *
from validation import *

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import *
from sklearn.pipeline import Pipeline

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

def week1(documents):
    n_total = len(documents)
    # #1
    helper.out('1-1.txt', n_total)
    print str(n_total) + ' reviews in total'

    # #2
    n_pos = len([d for d in documents if d[1] == 1])
    helper.out('1-2.txt', n_pos/float(n_total))
    print str(n_pos/float(n_total)) + ' of positive reviews'

    data = [' '.join(d[0]) for d in documents]
    target = [d[1] for d in documents]

    vectorizer = CountVectorizer()
    doc_term = vectorizer.fit_transform(data)

    # #3
    helper.out('3.txt', doc_term.shape[1])
    print str(doc_term.shape[1]) + ' features in total'

    classifier = LogisticRegression()

    pipe_logreg = Pipeline([('CountVectorizer', vectorizer), \
                            ('LogisticRegression', classifier)])

    accuracy_scores = validate(pipe_logreg, data, target)

    # #4
    print('Mean accuracy: %f' % accuracy_scores.mean())
    helper.out('4.txt', accuracy_scores.mean())

    roc_auc_scores = validate(pipe_logreg, data, target, scoring='roc_auc')

    # #5
    print('Mean roc_auc_score: %f' % roc_auc_scores.mean())
    helper.out('5.txt', roc_auc_scores.mean())


    classifier.fit(doc_term, target)
    features = vectorizer.get_feature_names()
    # print 'Features (%i): ' % len(features)
    #print features

    print 'Vocab (%i): ' % len(vectorizer.vocabulary_)
    print vectorizer.vocabulary_

    # print 'Coefs (%i):' % len(classifier.coef_[0])
    # print classifier.coef_

    n_best_feat = 2
    coeffs = np.array([abs(c) for c in classifier.coef_[0]])
    max_inds = coeffs.argsort()[-n_best_feat:][::-1]
    #print max_inds
    best_feats = [features[i] for i in max_inds]
    #print best_feats

    # #6
    print('Most important features: ' + str(best_feats))
    helper.out('6.txt', ' '.join(best_feats))

def week2(X, y):
    cnt_vectorizer = CountVectorizer()
    cnt_vectorizer.fit_transform(X)
    classifier = LogisticRegression()
    pipe_cnt_logreg = Pipeline([('countvectorizer', cnt_vectorizer),
                                ('logisticregression', classifier)])
    tf_vectorizer = TfidfVectorizer()
    pipe_tf_logreg = Pipeline([('tfidfvectorizer', tf_vectorizer),
                               ('logisticregression', classifier)])

    # #1
    scores = compare_accuracy([pipe_cnt_logreg, pipe_tf_logreg], X, y)
    helper.out('2-1.txt', [scores[0].mean(), scores[0].std(), scores[1].mean(), scores[1].std()])

    # #2
    cnt_vectorizer.min_df = 10
    scores_cnt_logreg_10 = np.array(cross_val_score(pipe_cnt_logreg, X, y, cv=n_cv))
    print 'Pipeline %s, min_df=10: accuracy mean = %f' % (pipe_cnt_logreg.named_steps.keys(), scores_cnt_logreg_10.mean())

    cnt_vectorizer.min_df = 50
    scores_cnt_logreg_50 = np.array(cross_val_score(pipe_cnt_logreg, X, y, cv=n_cv))
    print 'Pipeline %s, min_df=50: accuracy mean = %f' % (pipe_cnt_logreg.named_steps.keys(), scores_cnt_logreg_50.mean())
    helper.out('2-2.txt', [scores_cnt_logreg_10.mean(), scores_cnt_logreg_50.mean()])

    # #3
    pipe_cnt_logreg = Pipeline([('countvectorizer', cnt_vectorizer),
                                ('logisticregression', LogisticRegression())])
    pipe_cnt_svc = Pipeline([('countvectorizer', cnt_vectorizer),
                             ('linearsvc', LinearSVC())])
    pipe_cnt_sgd = Pipeline([('countvectorizer', cnt_vectorizer),
                             ('sgdclassifier', SGDClassifier())])
    scores = choose_classifier([pipe_cnt_logreg, pipe_cnt_svc, pipe_cnt_sgd], X, y)
    worst_score = min(scores.values())
    print 'Worst score: %f' % worst_score
    helper.out('2-3.txt',worst_score)

    # #4
    stop_words_dict = {
        'nltk stop-words': nltk.corpus.stopwords.words('english')[0],
        'sklearn stop-words': 'english'
    }
    scores = estimate_stop_words(stop_words_dict, classifier, X, y)
    helper.out('2-4.txt', scores.values())
