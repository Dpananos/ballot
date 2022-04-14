import os


from src.experiment import Experiment, load_data
from src.models import dummy_classifier, naive_bayes_classifier

import seaborn as sns
import matplotlib.pyplot as plt


def voting_experiment(file:str, label: str, save_dir: str) -> None:

    # Check that we have a place to store the results from the experiment
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(f"{save_dir}/classification_report")
        os.makedirs(f"{save_dir}/figures")

    # Load in the data
    X, y = load_data(file=file, label=label)

    _, ax = plt.subplots(dpi=240)
    ax.set_title("Byte Length Distribution By Vote")
    sns.histplot(x=X.ravel(), hue=y, ax=ax)
    ax.set_xlabel('Byte Length')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/conditional_distributions.png")

    # Load in the untrained models
    dummy_clf = dummy_classifier()
    nb_clf = naive_bayes_classifier()

    # Run experiments
    model_names = ("Dummy_Classifier", "Naive_Bayes")
    models = (dummy_clf, nb_clf)

    for name, model in zip(model_names, models):

        experiment = Experiment(experiment_name=name, model=model, save_dir=save_dir, X=X, y=y)
        experiment.run()
        experiment.validate()

    return None


def main() -> None:
    # Experiment 1: Just the mayor from Selwyn
    voting_experiment(file='selwyn-just-mayor.csv', label='mayor', save_dir='experiment_1')

    # Experiment 2
    voting_experiment(file='selwyn-mayor-councillor.csv', label='mayor', save_dir='experiment_2_mayor')
    voting_experiment(file='selwyn-mayor-councillor.csv', label='councillor', save_dir='experiment_2_councillor')

    # Experiment 3
    voting_experiment(file='ajax-ward-1.csv', label='mayor', save_dir='experiment_3_mayor')
    voting_experiment(file='ajax-ward-1.csv', label='councillor', save_dir='experiment_3_councillor')
    voting_experiment(file='ajax-ward-1.csv', label='regional_councillor', save_dir='experiment_3_regional_councillor')

    return None


if __name__ == "__main__":

    main()