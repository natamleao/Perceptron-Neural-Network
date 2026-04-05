import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate):
        self.learning_rate = learning_rate
        self._weights = np.random.uniform(-0.5, 0.5, (n_inputs,1))
        self._bias = np.random.uniform(-0.5,0.5)
        
    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> float:
        return self._bias
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value
        
    @weights.setter
    def weights(self, value: np.ndarray):
        self._weights = value
        
    @bias.setter
    def bias(self, value: float):
        self._bias = value

    def activation(self, v):
        return 1 if v >= 0 else -1

    def forward(self, x):
        v = np.dot(x, self._weights).item() + self._bias
        return self.activation(v)

    def update(self, x, error):
        self._weights += self.learning_rate * error * x.reshape(-1,1)
        self._bias += self.learning_rate * error
        
