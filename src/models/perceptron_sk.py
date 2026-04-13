from src.models.perceptron import Perceptron
from src.training.trainer import PerceptronTrainer
from src.prediction.predictor import PerceptronPredictor

class PerceptronSK:
    def __init__(self, learning_rate=0.1, max_epochs=1000, logger=None):
        self._learning_rate = learning_rate
        self._max_epochs = max_epochs
        self._logger = logger

        self._model = None
        self._trainer = None 
        
    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @property
    def logger(self) -> bool:
        return self._logger
    
    @property
    def model(self) -> Perceptron:
        return self._model
    
    @property
    def trainer(self) -> PerceptronTrainer:
        return self._trainer
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value
        
    @max_epochs.setter
    def max_epochs(self, value: int):
        self._max_epochs = value
        
    @logger.setter
    def logger(self, value: bool):
        self._logger = value

    @model.setter
    def model(self, value: Perceptron):
        self._model = value
        
    @trainer.setter
    def trainer(self, value: PerceptronTrainer):
        self._trainer = value

    def fit(self, X, y):
        n_inputs = X.shape[1]
        self.model = Perceptron(n_inputs, self.learning_rate)
        self.trainer = PerceptronTrainer()

        if y.ndim == 1: y = y.reshape(-1,1)

        self.trainer.train(self.model, X, y, self.max_epochs, self.logger)
        return self

    def predict(self, X):
        predictor = PerceptronPredictor()
        return predictor.predict(self.model, X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y.reshape(-1,1)).mean()
