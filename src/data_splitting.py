import pandas as pd 

def train_val_split(dataset:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide dataset en 80% entrenamiento y 20% validacion

        Parámetros de entrada:
            dataset(pd.DataFrame): el dataset a ser separado en set de entrenamiento y de validación 

        Parámetros de salida:
            train(pd.DataFrame): subset de entrenamiento
            validation(pd.DataFrame): subset de validación
    """
    shuffled = dataset.sample(frac=1,  random_state= 42).reset_index(drop=True)

    cut_idx = int(0.8* len(dataset)) #Split

    train = shuffled.iloc[:cut_idx].copy() #80%
    validation = shuffled.iloc[cut_idx:].copy() #20%
    print(f'Tamaño conjunto de entrenamiento -> {train.shape}, \nTamaño conjunto de validación -> {validation.shape}')

    return train, validation