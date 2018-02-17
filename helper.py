
import time
import datetime
import pickle
import pandas as pd
import numpy as np
now = datetime.datetime.now()

from sklearn.pipeline import Pipeline

def out(fname, s):
    with open(fname, 'w') as f:
        if isinstance(s, list):
            f.write(" ".join([str(el) for el in s]))
        else:
            f.write(str(s))
        f.close()

def write_predictions(fname, df):
    df.to_csv(fname, columns=['Id', 'y'], index=False)

def log_results(message, accuracy, filename='accuracies_products.txt'):
    with open(filename, 'a') as f:
        f.write('{}:\t{}\t{:f}\n'.format(now.strftime("%Y-%m-%d %H:%M"), message, accuracy))
        f.close()


def log_pipeline_results(pipeline, accuracy, filename='accuracies_products.txt'):
    with open(filename, 'a') as f:
        f.write('{}:\t{}\t{:f}\n'.format(now.strftime("%Y-%m-%d %H:%M"), pipeline.named_steps.keys(), accuracy))
        f.close()


def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper

def identity(x):
    return x

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def multiply_data(X, y, n):
    data = np.array(zip(X, y))
    X_ext = list(X)
    y_ext = list(y)
    samples = get_bootstrap_samples(data, n-1)
    for sample in samples:
        X_s = [s[0] for s in sample]
        y_s = [s[1] for s in sample]
        X_ext += X_s
        y_ext += y_s
    return X_ext, y_ext