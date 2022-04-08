import numpy as np
import pandas as pd

from src.experiment import Experiment

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

import seaborn as sns
import matplotlib.pyplot as plt


def simple_case(file)->None:

    # Load in the data
    df = pd.read_csv(file)
    df.columns = ["vote", "bytes"]

    _, ax = plt.subplots(dpi=240)
    ax.set_title("Byte Length Distribution By Vote")
    sns.histplot(data=df, x="bytes", hue="vote", ax=ax)
    plt.tight_layout()
    plt.savefig("figures/mayor_only_conditional_distributions.png")

    # Make data to pass to experiment
    X = df[["bytes"]].values
    y = df.vote.values
    
    # Split the data into a train/test set. Use half for training, half for testing.
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.5, random_state=19920908, stratify=y
    )

    # Baseline will be a dummy model
    dummy_clf = DummyClassifier(strategy="stratified")

    # Now for our model.
    # Here is how the prediction will go
    # When we receive a dataset, we will one hot encode (OHE) the Byte length
    # This allows for non-parametric estimation of P(Byte Lenfth|Vote)
    # In order to properly validate this model, we need to assume we have no knowledge of what the byte length can be
    # Create a pipeline which will automatically detect unique levels of Bytes and OHE them.
    # If at prediction time the OHE'er sees a new level, it should just ignore it.

    # Here is the OHE'er
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Here is an instantiation of the model we will be running.
    naive_bayes = MultinomialNB()

    # Place them together in a single estimator pipeline.  This will transform bytes and make predictions in a single call
    steps = [("ohe", one_hot_encoder), ("naive_bayes", naive_bayes)]

    nb_model = Pipeline(steps=steps)

    # Run experiments
    model_names = ("Mayor_Only_Dummy_Classifier", "Mayor_Only_Naive_Bayes")
    models = (dummy_clf, nb_model)
    for name, model in zip(model_names, models):

        experiment = Experiment(
            experiment_name=name,
            model=model,
            Xtrain=Xtrain,
            Xtest=Xtest,
            ytrain=ytrain,
            ytest=ytest,
        )

        experiment.run()

    return None

def complex_case(file, which="mayor")->None:

    # Load in the data
    df = pd.read_csv(file)
    df.columns = ["mayor", "councillor", "bytes"]

    _, ax = plt.subplots(dpi=240)
    ax.set_title("Byte Length Distribution By Vote")
    sns.histplot(data=df, x="bytes", hue=which, ax=ax)
    plt.tight_layout()
    plt.savefig("figures/mayor_complex_conditional_distributions.png")

    # Make data to pass to experiment
    X = df[["bytes"]].values
    y = df[which]
    
    # Split the data into a train/test set. Use half for training, half for testing.
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.5, random_state=19920908, stratify=y
    )

    # Baseline will be a dummy model
    dummy_clf = DummyClassifier(strategy="stratified")

    # Now for our model.
    # Here is how the prediction will go
    # When we receive a dataset, we will one hot encode (OHE) the Byte length
    # This allows for non-parametric estimation of P(Byte Lenfth|Vote)
    # In order to properly validate this model, we need to assume we have no knowledge of what the byte length can be
    # Create a pipeline which will automatically detect unique levels of Bytes and OHE them.
    # If at prediction time the OHE'er sees a new level, it should just ignore it.

    # Here is the OHE'er
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Here is an instantiation of the model we will be running.
    naive_bayes = MultinomialNB()

    # Place them together in a single estimator pipeline.  This will transform bytes and make predictions in a single call
    steps = [("ohe", one_hot_encoder), ("naive_bayes", naive_bayes)]

    nb_model = Pipeline(steps=steps)

    # Run experiments
    model_names = (f"{which}_Complex_Dummy_Classifier", f"{which}_Complex_Naive_Bayes")
    models = (dummy_clf, nb_model)
    for name, model in zip(model_names, models):

        experiment = Experiment(
            experiment_name=name,
            model=model,
            Xtrain=Xtrain,
            Xtest=Xtest,
            ytrain=ytrain,
            ytest=ytest,
        )

        experiment.run()

    return None


if __name__ == "__main__":

    simple_case(file="data/selwyn-just-mayor.csv")

    complex_case(file='data/selwyn-mayor-councillor.csv', which='mayor')
    complex_case(file='data/selwyn-mayor-councillor.csv', which='councillor')
