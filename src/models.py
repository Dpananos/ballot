from typing import Protocol
from .experiment import ScikitModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

class ScikitModel(Protocol):
    # Protocol for type hinting in the Experiment dataclass
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...


def dummy_classifier() -> ScikitModel:

    dummy_clf = DummyClassifier(strategy="stratified")

    return dummy_clf

def naive_bayes_classifier() -> ScikitModel:

    # Here is the OHE'er
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Here is an instantiation of the model we will be running.
    naive_bayes = MultinomialNB()

    # Place them together in a single estimator pipeline.  This will transform bytes and make predictions in a single call
    steps = [("ohe", one_hot_encoder), ("naive_bayes", naive_bayes)]

    nb_model = Pipeline(steps=steps)

    return nb_model