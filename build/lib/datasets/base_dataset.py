import pandas as pd

class Dataset:
    def __init__(self, path, target_column):
        self._path = path
        self._target_column = target_column
        
    @property 
    def path(self) -> str:
        return self._path
    
    @property
    def target_column(self) -> str:
        return self._target_column

    def load(self):
        df = pd.read_csv(self.path)

        X = df.drop(columns=[self.target_column]).values
        y = df[self.target_column].values.reshape(-1,1)

        return X, y