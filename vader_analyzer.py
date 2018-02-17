from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformations.base_transform import Transformer
import numpy as np

class SentimentAnalyzer(Transformer):
    def __init__(self):
        self._analyser = SentimentIntensityAnalyzer()

    def transform(self, X, y=None):
        Xt = [
            self.get_sentiment_scores(sentence) for sentence in X
        ]
        return np.squeeze(np.asarray(Xt))

    def get_sentiment_scores(self, sentence):
        snt = self._analyser.polarity_scores(sentence)
        return [snt['pos'],snt['neg']]

    def print_sentiment_scores(self, sentence):
        snt = self._analyser.polarity_scores(sentence)
        print("{:-<40} {}".format(sentence, str(snt)))