from typing import Protocol
from dataclasses import dataclass
from sklearn.model_selection import RepeatedKFold, cross_val_score
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
    Xtrain: ArrayLike
    ytrain: ArrayLike
    Xtest: ArrayLike
    ytest: ArrayLike

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
