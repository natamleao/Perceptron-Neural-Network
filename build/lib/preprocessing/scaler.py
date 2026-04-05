import numpy as np

class StandardScaler:
    def fit(self, X) -> None:
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
        
    @property
    def mean(self) -> float:
        return self._mean
    
    @property
    def std(self) -> float:
        return self._std

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean