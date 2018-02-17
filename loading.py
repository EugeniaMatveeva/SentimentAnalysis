import os
from xml.dom import minidom
import random, string

from nltk.corpus import movie_reviews
import pandas as pd


def load_moview_reviews(shuffle=True):
    X = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
    y = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]
    Xy = zip(X, y)
    if shuffle:
        random.shuffle(Xy, )
    else:
        random.shuffle(Xy, lambda: 0.42 )
    return [x[0] for x in Xy], [x[1] for x in Xy]


def load_product_reviews(path, sep='\t', header=None, shuffle=True):
    dt_train = pd.read_csv(path, sep=sep, header=header)
    X_train = dt_train.ix[:][0].values
    y_train = dt_train.ix[:][1].values
    Xy = zip(X_train, y_train)
    random.shuffle(Xy, lambda: random.random() if shuffle else 0.42)
    return [x[0] for x in Xy], [x[1] for x in Xy]


def load_product_reviews_test(path, sep='\t'):
    dt_test = pd.read_csv(path, sep=sep)
    return dt_test


def load_reviews(path, sep=',', shuffle=True, balance=True):
    df = pd.read_csv(path, sep=sep, encoding ='utf8')
    df = df.dropna()
    df = df.drop(df[df['text'].map(len) < 10].index)

    if balance:
        g = df.groupby('y')
        df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

    df.ix[:]['text'].apply(lambda x: x.replace('\n', ' '))
    X_train = df.ix[:]['text'].values
    y_train = df.ix[:]['y'].values
    Xy = zip(X_train, y_train)
    random.shuffle(Xy, lambda: random.random() if shuffle else 0.42)
    return [x[0] for x in Xy], [x[1] for x in Xy]


def load_reviews_xml(path):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    xmldoc = minidom.parse(path)
    itemlist = xmldoc.getElementsByTagName('review')
    reviews = [item.firstChild.nodeValue.encode('utf-8').decode('utf-8') for item in itemlist]
    df = pd.DataFrame(data={'text': reviews, 'Id': range(0, len(reviews))})
    return df
