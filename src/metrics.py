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

#Error absoluto medio
def mae(y_true:np.array,y_pred:np.array) -> float:
    """
    Computa el Error Medio Cuadrático entre el valor real y el predicho de cada muestra

        Parámetros de entrada:
            y_true(np.array): valores reales del target
            y_pred(np.array): valores estimados del target
        
        Parámetros de salida:
            (float): la raíz del MSE, siempre positiva
    """
    return np.mean(np.abs(y_true - y_pred))


#Coeficiente de determinacion R cuadrado
def r2(y_true, y_pred):
    """
    Computa el Coeficiente de determinacion R² entre el valor real y el predicho de cada muestra.
    Indica que proporcion de la varianza del target es explicada por el modelo

        Parámetros de entrada:
            y_true(np.array): valores reales del target
            y_pred(np.array): valores estimados del target
        
        Parámetros de salida:
            (float): el error cuadrático medio, siempre positivo
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_promedio = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_promedio)
    