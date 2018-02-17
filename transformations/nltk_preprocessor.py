import string

from collections import defaultdict
from inquirerlex_transform import InquirerLexTransform

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk import wordnet
synsets = wordnet.wordnet.synsets

from sklearn.base import BaseEstimator, TransformerMixin


# def load_data():
#     negids = movie_reviews.fileids('neg')
#     posids = movie_reviews.fileids('pos')
#     movie_reviews.sents
#
#     documents = [(list(cut_alien_text(movie_reviews.words(fileid))), 0) for fileid in negids] + \
#                 [(list(cut_alien_text(movie_reviews.words(fileid))), 1) for fileid in posids]
#     # documents = [(list(movie_reviews.words(fileid)), category)
#     #               for category in movie_reviews.categories()
#     #               for fileid in movie_reviews.fileids(category)]
#     return documents





class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """
    _corpus = defaultdict(str)

    def __init__(self, stopwords=set(sw.words('english')), punct=set(string.punctuation), lower=True, strip=True, add_PosNeg=True, replace_SynSet=True, filter_token=True):
        """
        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.
        """
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords
        self.punct      = punct
        self.lemmatizer = WordNetLemmatizer()
        self.add_PosNeg = add_PosNeg
        if add_PosNeg:
            self.inquirer_lex = InquirerLexTransform()
        self.replace_SynSet = replace_SynSet
        self.filter_token = filter_token
        self.include_words = {'no'}

    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        X_tokens = []
        for doc in X:
            tokens = list(self.tokenize(self.preprocess(doc)))
            if self.add_PosNeg:
                tokens += list(self.inquirer_lex._get_sentiment(tokens))
            X_tokens.append(tokens)
        return X_tokens

    def tokenize(self, document):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If punctuation or stopword, ignore token and continue
                # Leave only nouns, adverbs and adjectives
                if (self.stopwords and token in self.stopwords) \
                        or (self.punct and all(char in self.punct for char in token)) \
                        or (len(token) < 2 and self.punct and not all(char in self.punct for char in token)) \
                        or (self.filter_token and not self.filter_tag(tag, token))\
                        and token not in self.include_words:
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                if self.replace_SynSet:
                    # replace with synonyms from corpus if exist
                    yield self.add_to_corpus(lemma, tag)
                else:
                    yield lemma

    def filter_tag(self, tag, token):
        if (tag[:2] in ('NN', 'JJ', 'RB', 'VB')):
            '''('NN', 'JJ', 'RB', 'VB')'''
            return True
        return False

    def replace_with_Synset(self, w, tag):
        if (len(synsets(w)) > 0):
            return synsets(w)[0].name()
        return w

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = self.get_wordnet_tag(tag)
        return self.lemmatizer.lemmatize(token, tag)

    def add_to_corpus(self, word, tag):
        tag = self.get_wordnet_tag(tag)
        syns = self.get_syns(word, tag)
        for w in syns:
            if w in self._corpus:
                return w
        self._corpus[word] = tag
        return word

    @staticmethod
    def get_wordnet_tag(tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag
        """
        return {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

    @staticmethod
    def preprocess(sent):
        # alien_text = 'the flying inkpot rating system'
        # if alien_text in sent:
        #     return sent[:sent.find(alien_text)]
        return sent.replace(" '", "").replace("'", "")

    @staticmethod
    def get_syns(w, tag):
        ssets = synsets(w, tag)
        syns = set([l.name() for s in ssets for l in s.lemmas()])
        return syns