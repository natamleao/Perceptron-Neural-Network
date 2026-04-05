class PerceptronConfig:
    def __init__(self, learning_rate=0.1, max_epochs=1000):
        self._learning_rate = learning_rate
        self._max_epochs = max_epochs
        
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @property 
    def max_epochs(self) -> int:
        return self._max_epochs
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value
        
    @max_epochs.setter
    def max_epochs(self, value: int):
        self._max_epochs = value