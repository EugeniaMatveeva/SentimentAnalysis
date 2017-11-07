
import time
import datetime
now = datetime.datetime.now()

from sklearn.pipeline import Pipeline

def out(fname, s):
    with open(fname, 'w') as f:
        if isinstance(s, list):
            f.write(" ".join([str(el) for el in s]))
        else:
            f.write(str(s))
        f.close()

def log_pipeline_results(pipeline, accuracy):
    with open('accuracies.txt', 'a') as f:
        f.write('{}:]\t{}\t{:f}\n'.format(now.strftime("%Y-%m-%d %H:%M"), pipeline.named_steps.keys(), accuracy))
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