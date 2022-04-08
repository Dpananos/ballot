from typing import Protocol, List
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


class ScikitModel(Protocol):
    # Protocol for type hinting in the Experiment dataclass
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...

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

        self.Xtrain, self.Xtest, self.ytrain, self.ytest =  train_test_split(self.X, self.y, train_size=0.5, random_state=19920908, stratify=self.y)

    def run(self) -> None:

        self.model.fit(self.Xtrain, self.ytrain)
        ypred = self.model.predict(self.Xtest)
        cm = confusion_matrix(y_true=self.ytest, y_pred=ypred)

        # Make a classification report
        report_name = f"{self.save_dir}/results/{self.experiment_name}_classification_report.txt"
        with open(report_name, "w") as report:
            cr = classification_report(y_true=self.ytest, y_pred=ypred)
            report.write(cr)

        # Plot the confusion matix

        cm_fig, cm_ax = plt.subplots(dpi=240)
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
