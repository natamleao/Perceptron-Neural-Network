import numpy as np

class PerceptronPredictor:
    def predict(self, model, X):
        v = np.dot(X, model.weights) + model.bias

        return np.where(v >= 0, 1, -1)