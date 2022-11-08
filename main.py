import hydra
from omegaconf import OmegaConf
import logging
import pandas as pd
import numpy as np
from src.experiment import Experiment
from src.models import dummy_classifier, naive_bayes_classifier


def voting_experiment(cfg) -> None:

    df = pd.read_csv(cfg.dataset.file)
    X = df[cfg.dataset.features].values.reshape(-1, 1)
    y = df[cfg.dataset.target].values

    # Load in the untrained models
    if cfg.model.name == "Naive Bayes":
        model = naive_bayes_classifier()
    elif cfg.model.name == "Dummy Classifier":
        model = dummy_classifier()
    else:
        raise ValueError("Not Implemented")

    # Run experiments
    experiment = Experiment(model=model, X=X, y=y)
    logging.info("Experiment Results")
    logging.info(f"Model = {cfg.model.name}")
    logging.info('\n' + experiment.report)
        

    return None


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    voting_experiment(cfg)
    return None


if __name__ == "__main__":

    main()