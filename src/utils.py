import numpy as np

def estandarizar(X_train:np.ndarray, X_eval:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standarization of the data using the mean and standard deviation computed from the training set - > X_norm = (X - μ) / σ

        Entry parameters:
            X_train(np.ndarray): training feature matrix of shape (n_samples, n_samples)
            X_eval(np.ndarray): evaluation feature matrix of shape (n_samples, n_samples)

        Output parameters:
            X_train(np.ndarray): standarized training feature matrix of shape
            X_eval(np.ndarray): standarized test feature matrix of shape
            mean(np.ndarray): mean per feature computed from the training set
            std(np.ndarray): standard deviation per feature computed from the training set
    """
    mean = X_train.mean(axis = 0)
    
    std = X_train.std(axis = 0)
    #Replace the values where the std = 0 to avoid dividing by 0
    std = np.where(std == 0, 1, std)

    X_train = (X_train - mean) / std
    X_eval = (X_eval - mean) / std

    return X_train, X_eval, mean, std