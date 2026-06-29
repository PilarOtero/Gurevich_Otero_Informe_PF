import numpy as np

def estandarizar(X_train:np.ndarray, X_val:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estandarización de la información utilizando la media y el desvío estandar calculados sobre el set de entrenamiento -> X_normalizado = (X - μ) / σ

        Parámetros de entrada:
            X_train(np.ndarray): matriz de features de entrenamiento 
            X_val(np.ndarray): matriz de features de validación

        Parámetros de salida:
            X_train(np.ndarray): matriz de entrenamiento estandarizada 
            X_val(np.ndarray): matriz de validación estandarizada 
            mean(np.ndarray): media por feature obtenida del set de entrenamiento
            std(np.ndarray): desvío estandar por feature obtenido del set de entrenamiento
    """
    mean = X_train.mean(axis = 0)
    
    std = X_train.std(axis = 0)
    #Replace the values where the std = 0 to avoid dividing by 0
    std = np.where(std == 0, 1, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, mean, std