import pandas as pd
import numpy as np
from .models import ScikitModel
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import  make_scorer, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, log_loss
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

@dataclass
class Experiment:
    model: ScikitModel
    X: ArrayLike
    y: ArrayLike
    save_dir: str

    def run(self) -> None:
        
        Xtrain, Xtest, ytrain, ytest = train_test_split(self.X, self.y, random_state=0, train_size=0.5)

        self.model.fit(Xtrain, ytrain)
        ypred = self.model.predict(Xtest)

        report = classification_report(ytest, ypred)

        with open(self.save_dir, 'w') as file:
            file.write(report)


        return None
