import pandas as pd
import os

class XORDataset:
    def __init__(self, path='data/raw/xor.csv'):
        self.path = path

    def generate(self):
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        data = [
            [ 1,  1, -1],
            [ 1, -1,  1],
            [-1,  1,  1],
            [-1, -1, -1]
        ]

        df = pd.DataFrame(data, columns=['x1','x2','class'])

        df.to_csv(self.path, index=False)