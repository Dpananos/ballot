import pandas as pd
import numpy as np

import logging
import hydra
from omegaconf import OmegaConf

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def voting_experiment(cfg) -> None:

    df = pd.read_csv(cfg.dataset.file)
    X = df[cfg.dataset.features].values.reshape(-1, 1)
    y = df[cfg.dataset.target].values

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.5, random_state=0)

    # Load in the untrained models
    if cfg.model.name == "Naive Bayes":

        model = Pipeline([
            ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore')),
            ('nb', MultinomialNB())
        ])
                
    elif cfg.model.name == "Dummy Classifier":
        model = DummyClassifier(strategy="uniform")

    else:
        raise ValueError("Not Implemented")

    # Run experiments
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    score = accuracy_score(ytest, ypred)
    
    logging.info("Experiment Results")
    logging.info(f"Model = {cfg.model.name}")
    logging.info(f'Accuracy Score: {score:.2f}')
    
    return None


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    voting_experiment(cfg)
    return None


if __name__ == "__main__":

    main()