import hydra
import pandas as pd
import numpy as np
from src.experiment import Experiment
from src.models import dummy_classifier, naive_bayes_classifier


def voting_experiment(cfg) -> None:

    df = pd.read_csv(cfg.paths.data)
    X = df[cfg.params.features].values.reshape(-1, 1)
    y = df[cfg.params.target].values

    # Load in the untrained models
    dummy_clf = dummy_classifier()
    nb_clf = naive_bayes_classifier()

    # Run experiments
    model_names = ("Dummy_Classifier", "Naive_Bayes")
    models = (dummy_clf, nb_clf)

    for name, model in zip(model_names, models):

        experiment = Experiment(model=model, save_dir=cfg.paths.results, X=X, y=y)
        experiment.run()
        print('done')

    return None


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg) -> None:
    voting_experiment(cfg)
    return None


if __name__ == "__main__":

    main()