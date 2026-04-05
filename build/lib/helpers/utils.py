import numpy as np

def train_test_split_manual(X, y, test_size=0.2, shuffle=True, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx].reshape(-1,1)
    y_test = y[test_idx].reshape(-1,1)
    
    return X_train, X_test, y_train, y_test