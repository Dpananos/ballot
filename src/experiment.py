from dataclasses import dataclass, field
from typing import Union, List

from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

@dataclass
class Experiment:

    experiment_name: str
    model: Union[Pipeline, DummyClassifier]
    Xtrain: ArrayLike
    ytrain: ArrayLike
    Xtest: ArrayLike
    ytest: ArrayLike
    accuracy_score: float = field(init=False)
    stratified_accuracy: ArrayLike = field(init=False)


    def __post_init__(self) -> None:

        accuracy, stratified_accuracy = self.run()
        self.accuracy_score = accuracy
        self.stratified_accuracy = stratified_accuracy


    def run(self)->List[float]:

        self.model.fit(self.Xtrain, self.ytrain)
        ypred = self.model.predict(self.Xtest)
        cm = confusion_matrix(y_true=self.ytest, y_pred=ypred)

        # Compute accuracy scores

        accuracy = accuracy_score(y_true=self.ytest, y_pred=ypred)
        stratified_accuracy = cm.diagonal()/cm.sum(axis=1)

        # Make a classification report
        with open(f'results/{self.experiment_name}_classification_report.txt', 'w') as report:
            cr = classification_report(y_true = self.ytest, y_pred=ypred)
            report.write(cr)

        # Plot the confusion matix

        cm_fig, cm_ax = plt.subplots(dpi=240)
        ConfusionMatrixDisplay.from_predictions(y_true=self.ytest, y_pred=ypred, ax=cm_ax, xticks_rotation=45,  normalize='true')
        plt.tight_layout()
        plt.savefig(f'figures/{self.experiment_name}_conusion_matrix.png')

        return [accuracy, stratified_accuracy]





