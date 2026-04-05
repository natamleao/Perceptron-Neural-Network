import numpy as np
import pandas as pd
import os

class FruitDataset:
    def __init__(self, path='data/raw/fruit.csv'):
        self.path = path

    def generate(self):
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # Maçãs
        doce_macas = np.random.uniform(6, 9, 200)
        acidez_macas = np.random.uniform(2, 4.5, 200)
        macas = np.column_stack((doce_macas, acidez_macas))

        # Laranjas
        doce_laranjas = np.random.uniform(8, 10, 200)
        acidez_laranjas = np.random.uniform(5, 7, 200)
        laranjas = np.column_stack((doce_laranjas, acidez_laranjas))

        # Juntar dados e classes
        X = np.vstack((macas, laranjas))
        y = np.hstack((np.ones(200), -np.ones(200)))

        data = np.hstack((X, y.reshape(-1,1)))

        # Criar DataFrame
        df = pd.DataFrame(data, columns=['sweetness','acidity','class'])

        # Salvar CSV
        df.to_csv(self.path, index=False)