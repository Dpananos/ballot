from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

def dummy_classifier():

    dummy_clf = DummyClassifier(strategy="stratified")

    return dummy_clf


def naive_bayes_classifier():

    # Here is the OHE'er
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Here is an instantiation of the model we will be running.
    naive_bayes = MultinomialNB()

    # Place them together in a single estimator pipeline.  This will transform bytes and make predictions in a single call
    steps = [("ohe", one_hot_encoder), ("naive_bayes", naive_bayes)]

    nb_model = Pipeline(steps=steps)

    return nb_model