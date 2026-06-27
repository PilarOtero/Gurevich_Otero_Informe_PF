import numpy as np
import pandas as pd
import torch

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
    Computa el Error Medio Absoluto entre el valor real y el predicho de cada muestra

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
    

def metricas_train_val(modelo, X_train, y_train, X_val, y_val, categorico=False):

    X_train_eval = X_train.copy()
    X_val_eval = X_val.copy()

    if categorico:
        columnas = X_train_eval.select_dtypes(include=["object","category"]).columns

        for col in columnas:
            X_train_eval[col] = X_train_eval[col].astype("category")
            X_val_eval[col] = pd.Categorical(
                X_val_eval[col],
                categories=X_train_eval[col].cat.categories
            )

    pred_train = modelo.predict(X_train_eval)
    pred_val = modelo.predict(X_val_eval)

    return {
        "RMSE Train": rmse(y_train, pred_train),
        "RMSE Val": rmse(y_val, pred_val),
        "MAE Train": mae(y_train, pred_train),
        "MAE Val": mae(y_val, pred_val),
        "R² Train": r2(y_train, pred_train),
        "R² Val": r2(y_val, pred_val)
    }



def metricas_train_val_nn(modelo, X_train, y_train, X_val, y_val):

    device = next(modelo.parameters()).device
    modelo.eval()

    with torch.no_grad():
        X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(device)
        X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(device)

        pred_train_log = modelo(X_train_t).cpu().numpy()
        pred_val_log = modelo(X_val_t).cpu().numpy()

    pred_train = np.expm1(pred_train_log)
    pred_val = np.expm1(pred_val_log)

    pred_train = np.clip(pred_train, 0, None)
    pred_val = np.clip(pred_val, 0, None)

    return {
        "RMSE Train": rmse(y_train, pred_train),
        "RMSE Val": rmse(y_val, pred_val),
        "MAE Train": mae(y_train, pred_train),
        "MAE Val": mae(y_val, pred_val),
        "R² Train": r2(y_train, pred_train),
        "R² Val": r2(y_val, pred_val)
    }