import numpy as np

#Error cuadrático medio
def mse(y_true:np.array,y_pred:np.array) -> float:
    """
    Computa el Error Cuadrático Medio entre el valor real y el predicho de cada muestra

        Parámetros de entrada:
            y_true(np.array): valores reales del target
            y_pred(np.array): valores estimados del target
        
        Parámetros de salida:
            (float): el error cuadrático medio, siempre positivo
    """

    return np.mean((y_true - y_pred)**2)

#Raíz cuadrada del MSE
def rmse(y_true:np.array,y_pred:np.array) -> float:
    """
        Computa la raíz cuadrada del Error Cuadrático Medio entre el valor real y el predicho de cada muestra

        Parámetros de entrada:
            y_true(np.array): valores reales del target
            y_pred(np.array): valores estimados del target
        
        Parámetros de salida:
            (float): la raíz del MSE, siempre positiva
    """
    return np.sqrt(mse(y_true,y_pred))

def r2(y_true, y_pred):
    pass