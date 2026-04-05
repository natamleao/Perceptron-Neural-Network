import os
from src.datasets.boolean_dataset_and import ANDDataset
from src.datasets.boolean_dataset_or import ORDataset
from src.datasets.boolean_dataset_xor import XORDataset
from src.datasets.fruit_dataset import FruitDataset
from src.datasets.vehicle_dataset import VehicleDataset
from src.datasets.region_dataset import RegionDataset

from src.datasets.base_dataset import Dataset
from src.models.perceptron_sk import PerceptronSK
from src.training.logger import TrainingLogger
from src.preprocessing.scaler import StandardScaler
from src.visualization.decision_boundary import DecisionBoundaryPlotter
from src.helpers.utils import train_test_split_manual

datasets = {
    'and': ANDDataset,
    'or': ORDataset,
    'xor': XORDataset,
    'fruit': FruitDataset,
    'vehicle': VehicleDataset,
    'region': RegionDataset
}

def generate_datasets():
    print('+' + 78*'-' + '+')
    print('+' + 30*'-' + ' Gerando datasets ' + 30*'-' + '+')
    print('+' + 78*'-' + '+')

    os.makedirs('data/raw', exist_ok=True)

    for name, cls in datasets.items():

        path = f'data/raw/{name}.csv'

        dataset = cls(path=path)
        dataset.generate()

        print(f'+ {name.upper()} gerado em {path}')
        print('+' + 78*'-' + '+')


def train_perceptron():
    print('+' + 28*'-' + ' Treinando perceptron ' + 28*'-' + '+')

    for name in datasets.keys():

        print('+' + 78*'-' + '+')
        print(f'+ Dataset: {name.upper()}')

        path = f'data/raw/{name}.csv'

        dataset = Dataset(path, target_column='class')
        X, y = dataset.load()
        
        scaler = StandardScaler()
        
        if name not in ['and', 'or', 'xor']:
            X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_seed=42)     

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        X_train = X_test = scaler.fit_transform(X)
        y_train = y_test = y

        logger = TrainingLogger(enabled=True)

        model = PerceptronSK(learning_rate=0.1, max_epochs=1000, logger=logger)

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        print(f'+ Acurácia: {accuracy*100:.2f}%')

        if X.shape[1] == 2:
            plotter = DecisionBoundaryPlotter()
            plotter.plot(model.model, scaler, X_test, y_test)

        else:
            print('+ Dataset não é 2D, não é possível plotar.')
            
def main():

    print('\n+' + 78*'-' + '+')
    print('+ Perceptron Experiments')
    print('+' + 78*'-' + '+\n')

    generate_datasets()
    train_perceptron()

    print('\n+' + 78*'-' + '+')
    print('Experimentos finalizados.')
    print('+' + 78*'-' + '+\n')


if __name__ == '__main__':
    main()