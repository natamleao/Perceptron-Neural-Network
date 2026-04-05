import numpy as np
import pandas as pd
import os

class RegionDataset:
    def __init__(self, path='data/raw/region.csv'):
        self.path = path 

    def generate(self):
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # Região Norte (1)
        lat_norte = np.random.uniform(0.0, 50.0, 200)
        lon_norte = np.random.uniform(0.0, 50.0, 200)
        norte = np.column_stack((lat_norte, lon_norte))

        # Região Sul (-1)
        lat_sul = np.random.uniform(25.0, 75.0, 200)
        lon_sul = np.random.uniform(25.0, 75.0, 200)
        sul = np.column_stack((lat_sul, lon_sul))
        
        # Juntar os dados
        X = np.vstack((norte, sul))
        y = np.hstack((np.ones(200), -np.ones(200)))
        
        data = np.hstack((X, y.reshape(-1,1)))

        # Criar DataFrame
        df = pd.DataFrame(data, columns=['latitude','longitude','class'])

        # Salvar CSV
        df.to_csv(self.path, index=False)