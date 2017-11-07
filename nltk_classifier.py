import helper

import random
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class NltkClassifier():

    def __init__(self, documents):
        self._max_features = 2000

        self._documents = documents
        words = [w for doc, _ in documents for w in doc]
        self._all_words = nltk.FreqDist(w.lower() for w in words)
        data = [' '.join(d[0]) for d in documents]
        self._count_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2), max_features=self._max_features)
        self._count_matrix = self._count_vectorizer.fit_transform(data)
        self._tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=self._max_features)

    def fit(self, train_set, test_set):
        self._classifier = nltk.NaiveBayesClassifier.train(train_set)
        acc = nltk.classify.accuracy(self._classifier, test_set)

        return acc

    def contain_features(self, document):
        document_words = set(document)
        word_features = list(self._all_words)[:self._max_features]

        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    def count_features(self, document):
        features = {}
        for word, ix in self._count_vectorizer.vocabulary_.iteritems():
            features['count({})'.format(word)] = self._count_matrix[:, ix]
        return features

    def tfidf_features(self, document):
        cnt_vectorizer = TfidfVectorizer(min_df=2)
        document_words = set(document)
        word_features = list(self._all_words)[:]

        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    def get_errors(self, test_data):
        errors = []
        for (doc, tag) in test_data:
            guess = self._classifier.classify(self.contain_features(doc))
            if guess != tag:
                errors.append((tag, guess, doc))
        return errors

    def test_classifier(self):
        random.shuffle(self._documents)
        featuresets = [(self.count_features(d), c) for (d, c) in self._documents]
        train_set, test_set = featuresets[100:], featuresets[:100]
        acc = self.fit(train_set, test_set)
        helper.save_accuracy('NaiveBayesClassifier, bit contains feature array', acc)
        print(acc)
        self._classifier.show_most_informative_features(10)

        # test_data = self._documents[:100]
        # errors = self.get_errors(test_data)
        # for (tag, guess, doc) in errors[:10]:
        #     print "a=%i: y=%i, x=%s" % (tag, guess, " ".join(doc))

