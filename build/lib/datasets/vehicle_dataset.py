import numpy as np
import pandas as pd
import os

class VehicleDataset:
    def __init__(self, path='data/raw/vehicle.csv'):
        self.path = path 

    def generate(self):
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # Sedan compacto (1)
        x1_compacto = np.random.uniform(1.62, 1.77, 200)
        x2_compacto = np.random.uniform(2.40, 2.59, 200)
        compacto = np.column_stack((x1_compacto, x2_compacto))

        # Sedan médio (-1)
        x1_medio = np.random.uniform(1.78, 1.92, 200)
        x2_medio = np.random.uniform(2.62, 2.86, 200)
        medio = np.column_stack((x1_medio, x2_medio))

        # Juntar os dados
        X = np.vstack((compacto, medio))
        y = np.hstack((np.ones(200), -np.ones(200)))
        
        data = np.hstack((X, y.reshape(-1,1)))

        # Criar DataFrame
        df = pd.DataFrame(data, columns=['width','wheelbase','class'])

        # Salvar CSV
        df.to_csv(self.path, index=False)