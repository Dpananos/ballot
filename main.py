import os
import pandas as pd

from src.experiment import Experiment
from src.models import dummy_classifier, naive_bayes_classifier

import seaborn as sns
import matplotlib.pyplot as plt


def voting_experiment(file:str, labels: str, features: str, save_dir: str) -> None:

    # Check that we have a place to store the results from the experiment
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(f"{save_dir}/classification_report")
        os.makedirs(f"{save_dir}/figures")

    # Load in the data
    df = pd.read_csv(file)

    _, ax = plt.subplots(dpi=240)
    ax.set_title("Byte Length Distribution By Vote")
    sns.histplot(data=df, x=features, hue=labels, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/conditional_distributions.png")

    # Make data to pass to experiment
    X = df[[features]].values
    y = df[labels].values

    # Load in the untrained models
    dummy_clf = dummy_classifier()
    nb_clf = naive_bayes_classifier()

    # Run experiments
    model_names = ("Dummy_Classifier", "Naive_Bayes")
    models = (dummy_clf, nb_clf)

    for name, model in zip(model_names, models):

        experiment = Experiment(experiment_name=name, model=model, save_dir=save_dir, X=X, y=y)
        experiment.run()

    return None


def main() -> None:

    voting_experiment(file='data/selwyn-just-mayor.csv', labels='mayor', features='byte_length', save_dir='mayor_only')
    voting_experiment(file='data/selwyn-mayor-councillor.csv', labels='mayor', features='byte_length', save_dir='mayor_councillor_mayor')
    voting_experiment(file='data/selwyn-mayor-councillor.csv', labels='councillor', features='byte_length', save_dir='mayor-councillor_councillor')

    return None


if __name__ == "__main__":

    main()