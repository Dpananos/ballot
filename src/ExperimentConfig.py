from dataclasses import dataclass

@dataclass
class Dataset:
    name: str
    file: str
    features: str
    target: str

@dataclass
class Model:
    name: str

@dataclass
class ExperimentConfig:

    dataset: Dataset 
    model: Model 