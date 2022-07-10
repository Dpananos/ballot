import pandas as pd
import numpy as np
from .models import ScikitModel
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import  make_scorer, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, log_loss
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


def load_data(file:str, label:str, root:str ='data'):

    data_file = Path(f'{root}/{file}')

    if not data_file.exists():
        raise ValueError('This file does not exist.')
    
    df = pd.read_csv(data_file)

    X = df.loc[:, ['byte_length']].values
    y = df.loc[:, label].values

    return X, y

@dataclass
class Experiment:

    experiment_name: str
    model: ScikitModel
    save_dir: str
    X: ArrayLike
    y: ArrayLike
    Xtrain: ArrayLike = field(init=False)
    ytrain: ArrayLike = field(init=False)
    Xtest: ArrayLike  = field(init=False)
    ytest: ArrayLike  = field(init=False)

    def __post_init__(self) -> None:

        self.Xtrain, self.Xtest, self.ytrain, self.ytest =  train_test_split(self.X, self.y, train_size=0.5, random_state=1010)

    def validate(self):

        metrics = {
            'accuracy': make_scorer(accuracy_score), 
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'logloss': make_scorer(log_loss, needs_proba=True, normalize=True, labels = np.unique(self.y))
        }

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=19920908)

        results = cross_validate(self.model, self.X, self.y, cv=cv, scoring=metrics, return_train_score=False)

        results_df = pd.DataFrame(results)

        results_df.to_csv(f'{self.save_dir}/{self.experiment_name}_cross_validated_performance.csv')


    def run(self) -> None:

        self.model.fit(self.Xtrain, self.ytrain)
        ypred = self.model.predict(self.Xtest)

        # Make a classification report
        report_name = f"{self.save_dir}/classification_report/{self.experiment_name}_classification_report.txt"
        with open(report_name, "w") as report:
            cr = classification_report(y_true=self.ytest, y_pred=ypred)
            report.write(cr)

        # Plot the confusion matix
        _, cm_ax = plt.subplots(dpi=240)
        ConfusionMatrixDisplay.from_predictions(
            y_true=self.ytest,
            y_pred=ypred,
            ax=cm_ax,
            xticks_rotation=45,
            normalize="true",
        )
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/figures/{self.experiment_name}_conusion_matrix.png")

        return None
