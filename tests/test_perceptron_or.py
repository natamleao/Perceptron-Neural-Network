from src.datasets.base_dataset import Dataset
from src.models.perceptron_sk import PerceptronSK
from src.preprocessing.scaler import StandardScaler 
from src.datasets.boolean_dataset_or import ORDataset
from src.training.logger import TrainingLogger

def test_perceptron_or_dataset():

    path = "data/test/or_test.csv"

    dataset_gen = ORDataset(path=path)
    dataset_gen.generate()

    dataset = Dataset(path, target_column="class")
    X, y = dataset.load()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logger = TrainingLogger(enabled=False)

    model = PerceptronSK(learning_rate=0.1, max_epochs=1000, logger=logger)

    model.fit(X, y)

    accuracy = model.score(X, y)

    assert accuracy == 1.0