import pickle
from operator import itemgetter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report as clsr, accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer

from helper import *
from parameter_tuning import evaluate, grid_search
from transformations.nltk_preprocessor import NLTKPreprocessor
from transformations.inquirerlex_transform import InquirerLexTransform

now = datetime.datetime.now()

class Evaluator:

    def __init__(self, preprocessor=None, vectorizer=None, use_inqlex=False, load_path=None):
        if load_path is not None:
            self.load_data_transformation(load_path)
            self.model_is_fit = True
        else:
            self.model_is_fit = False
            self.preprocessor = preprocessor or NLTKPreprocessor()
            self.vectorizer = vectorizer or TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), min_df=2)
        self.use_lex = use_inqlex

    def get_step(self, step_name):
        return self.model.named_steps[filter(lambda nm: step_name in nm, self.model.named_steps)[0]]

    def prepare_data_transformation(self, X, y, save_path=None):
        X_prep = self.preprocessor.fit_transform(X, y)
        X_vect = self.vectorizer.fit_transform(X_prep, y)
        self.save(X_vect, save_path + '_dt_matrix')
        self.save(self.preprocessor, save_path + '_preprocessor')
        self.save(self.vectorizer, save_path + '_vectorizer')

    def load_data_transformation(self, load_path):
        self.preprocessor = self.load(load_path + '_preprocessor')
        self.vectorizer = self.load(load_path + '_vectorizer')
        self.dt_matrix = self.load(load_path + '_dt_matrix')

    def test_classifier(self, classifier, classifier_name, X, y, test_split=0, show_features=False, show_errors=False):
        # Label encode the targets
        self.labels = LabelEncoder()
        y = self.labels.fit_transform(y)

        print("Evaluation model fit")
        if test_split > 0:
            X_train, X_test, y_train, y_test = tts(X, y, test_size=test_split, random_state=42)
            print("Building for evaluation")
            self.build_model(classifier, classifier_name)
            self.fit_model(X_train, y_train)
            y_pred, score = self.evaluate_on_test_split(X_test, y_test)

            if show_features and hasattr(classifier, 'coef_'):
                print(self.show_model_output())
            if show_errors:
                self.show_error_predictions(y_pred, X_test, y_test)
        else:
            self.build_model(classifier, classifier_name)
            score = self.evaluate_on_cross_val(X, y, classifier, classifier_name)
        log_pipeline_results(self.model, score)

    def build_model(self, classifier, classifier_name):
        if self.use_lex:
            vectorizer_name = type(self.vectorizer).__name__ + '+InquirerLexTransform'
            lex_vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)
            pipeline2 = make_pipeline(InquirerLexTransform(), lex_vectorizer)
            ext = [self.vectorizer, pipeline2]
            self.vectorizer = make_union(*ext)
            self.model = Pipeline([
                ('preprocessor:' + type(self.preprocessor).__name__, self.preprocessor),
                ('vectorizer:' + vectorizer_name, self.vectorizer),
                ('normalizer:', Normalizer()),
                ('classifier:' + classifier_name or type(classifier).__name__, classifier),
            ])
        else:
            self.model = Pipeline([
                ('preprocessor:' + type(self.preprocessor).__name__, self.preprocessor),
                ('vectorizer:' + type(self.vectorizer).__name__, self.vectorizer),
                ('classifier:' + classifier_name or type(classifier).__name__, classifier),
            ])

    def fit_model(self, X, y):
        if self.model_is_fit:
            self.get_step('classifier').fit(self.dt_matrix, y)
        else:
            self.model.fit(X, y)
        self.model.labels_ = self.labels

    def evaluate_on_cross_val(self, X, y, classifier, classifier_name=None):
        score = evaluate(self.model, X, y, 3)
        print("Model mean accuracy: {:f}".format(score))
        return score

    def evaluate_on_test_split(self, X_test, y_test, verbose=True):
        if verbose:
            print("Classification Report:\n")

        y_pred = self.model.predict(X_test)
        print(clsr(y_test, y_pred))
        score = accuracy_score(y_test, y_pred)
        return y_pred, score

    def choose_classifier_parameters(self, classifier, classifier_name, X, y, params_grid):
        self.build_model(classifier, classifier_name)
        res = grid_search(self.model, params_grid, X, y)
        print ('Best params: ', res.best_params_)
        print 'Best score: %f' % res.best_score_
        return res


    def save_model(self, save_path, steps):
        for s in steps:
            self.save(self.get_step(s), save_path + '_' + s)

    def save(self, obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_feature_vectors(self, text):
        # Extract the vectorizer and the classifier from the pipeline
        preprocessor = self.get_step('preprocessor')
        vectorizer = self.get_step('vectorizer')
        classifier = self.get_step('classifier')

        # Check to make sure that we can perform this computation
        if not hasattr(classifier, 'coef_'):
            raise TypeError(
                "Cannot compute most informative features on {} model.".format(
                    classifier.__class__.__name__
                )
            )
        if text is not None:
            # Compute the coefficients for the text
            tprep = preprocessor.transform([text])
            tvec = vectorizer.transform(tprep).toarray()
        else:
            # Otherwise simply use the coefficients
            tvec = classifier.coef_
        return tprep, tvec

    def show_model_output(self, text=None, y=None):
        tprep, tvec = self.get_feature_vectors(text)

        # Create the output string to return
        output = []

        # If text, add the predicted value to the output.
        if text is not None:
            output.append("\"{}\"".format(text))
            output.append("\nPreprocessed: {}\n".format(" ".join(tprep[0])))
            output.append("Classified as: {}".format(self.model.predict([text])[0]))
            if y is not None:
                output.append("Correct label is: {}".format(y))
            output.append("")

        return "\n".join(output)

    def show_model_features(self, text=None, n=20):
        """
        Computes the n most informative features of the model. If text is given, then will
        compute the most informative features for classifying that text.
        Note that this function will only work on linear models with coefs_
        """
        vectorizer = self.get_step('vectorizer')
        tprep, tvec = self.get_feature_vectors(text)

        output = []
        # Zip the feature names with the coefs and sort
        coefs = filter(
            lambda v: abs(v[0]) > 1e-10,
            sorted(
                zip(tvec[0], vectorizer.get_feature_names()), key=itemgetter(0), reverse=True
            )
        )
        topn  = zip(coefs[:n], coefs[:-(n+1):-1])
        # Create two columns with most negative and most positive features.
        for (cp, fnp), (cn, fnn) in topn:
            output.append(
                "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
            )
        return "\n".join(output)


    def show_transformed_texts(self, texts, preprocessor):
        ttexts = preprocessor.transform(texts)
        for t, tt in zip(texts, ttexts):
            print t
            print tt

    def show_error_predictions(self, y_pred, X_test, y_test):
        error_indices = [i for i, yi in enumerate(y_test) if y_pred[i] != y_test[i]]

        for ind in error_indices:
            print(self.show_model_output(text=X_test[ind], y=y_test[ind]))

