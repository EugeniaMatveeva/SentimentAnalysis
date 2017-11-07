import pickle
from operator import itemgetter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report as clsr, accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer

from helper import *
from parameter_tuning import evaluate
from transformations.nltk_preprocessor import NLTKPreprocessor
from transformations.inquirerlex_transform import InquirerLexTransform

now = datetime.datetime.now()

class Evaluator:

    def __init__(self, preprocessor=None, vectorizer=None, use_test_split=True, use_inqlex=False, save_paths=None, load_paths=None):
        if load_paths and 'preprocessor' in load_paths:
            self.preprocessor = self.load_step(save_paths['preprocessor'])
            self.preprocessor_is_fit = True
        else:
            self.preprocessor = preprocessor or NLTKPreprocessor()
        if load_paths and 'vectorizer' in load_paths:
            self.vectorizer = self.load_step(save_paths['vectorizer'])
            self.vectorizer_is_fit = True
        else:
            self.vectorizer = vectorizer or TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), min_df=2)
            self.vectorizer_is_fit = False
        self.use_test_split = use_test_split
        self.use_lex = use_inqlex
        self.save_paths = save_paths

    def get_step(self, step_name):
        return self.model.named_steps[filter(lambda nm: step_name in nm, self.model.named_steps)[0]]

    def test_classifier(self, classifier, classifier_name, X, y, show_features=False, show_errors=False):
        # Label encode the targets
        self.labels = LabelEncoder()
        y = self.labels.fit_transform(y)

        print("Evaluation model fit")
        if self.use_test_split:
            X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
            print("Building for evaluation")
            self.build_classifier(classifier, classifier_name, X_train, y_train)
            y_pred, score = self.evaluate_on_test_split(X_test, y_test)
            if show_features and hasattr(classifier, 'coef_'):
                print(self.show_model_output())

            if show_errors:
                self.show_error_predictions(y_pred, X_test, y_test)
        else:
            score = self.evaluate_on_cross_val(X, y, classifier, classifier_name)
        log_pipeline_results(self.model, score)

        if self.save_paths:
            for path in self.save_paths:
                if 'preprocessor' in self.save_paths: self.save_step('preprocessor', self.save_paths['preprocessor'])
                if 'vectorizer' in self.save_paths: self.save_step('vectorizer', self.save_paths['vectorizer'])
                if 'model' in self.save_paths: self.save_step('model', self.save_paths['model'])


    def build_classifier(self, classifier, classifier_name, X, y):
        if (self.use_lex):
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
        self.model.fit(X, y)
        self.model.labels_ = self.labels

    def evaluate_on_cross_val(self, X, y, classifier, classifier_name=None):
        self.build_classifier(classifier, classifier_name, X, y)
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


    def save_step(self, step_name, path=None):
        step = self.model.named_steps[filter(lambda nm: step_name in nm, self.model.named_steps)[0]]
        with open(path or step_name, 'wb') as f:
            pickle.dump(step, f)

    def load_step(self, path):
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

