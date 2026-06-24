import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from src.metrics import r2, rmse, mae

def entrenar_regresion_lineal(X_train:pd.DataFrame, y_train:pd.Series, X_val:pd.DataFrame, y_val:pd.Series, modelo = None, print_ = False) -> tuple:
    """
    Entrena un modelo de regresión lineal (o una variante regularizada como Ridge/Lasso si se pasa por parámetro) y evalúa su desempeño sobre el set de validación.

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features de entrenamiento
            y_train(pd.Series): target de entrenamiento
            X_val(pd.DataFrame): matriz de features de validación
            y_val(pd.Series): target de validación
            modelo(sklearn estimator): modelo a entrenar

        Parámetros de salida:
            modelo: modelo entrenado
            y_pred(np.ndarray): predicciones sobre X_val
            rmse_score(float): raíz del error cuadrático medio sobre validación
            r2_score(float): coeficiente de determinación sobre validación
            mae_score(float): error absoluto medio sobre validación
    """
    if modelo is None:
        modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)

    #Métricas
    rmse_score = rmse(y_val, y_pred)
    r2_score = r2(y_val, y_pred)
    mae_score = mae(y_val, y_pred)

    if print_:
        print(f'RMSE = {rmse_score:.2f}')
        print(f'MAE = {mae_score:.2f}')
        print(f'R² = {r2_score:.2f}')

    return modelo, y_pred, rmse_score, r2_score, mae_score

def definir_regularizacion(X_train:pd.DataFrame, y_train:pd.Series, X_val:pd.DataFrame, y_val:pd.Series, lambdas:list[float]) -> pd.DataFrame:
    """
    Compara distintos modelos de regresión regularizada (Ridge y Lasso) entrenando uno por cada valor de alpha provisto, para encontrar la combinación con mejor desempeño.

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features de entrenamiento
            y_train(pd.Series): target de entrenamiento
            X_val(pd.DataFrame): matriz de features de validación
            y_val(pd.Series): target de validación
            alphas(list of float): lista de valores de alpha a probar

        Parámetros de salida:
            (pd.DataFrame): resultados de todas las combinaciones Modelo/Alpha, ordenado de mayor a menor R² sobre validación
    """
    resultados = []

    for lambda_ in lambdas:
        for nombre, modelo in [('Ridge', Ridge(alpha = lambda_, solver = 'svd')), ('Lasso', Lasso(alpha = lambda_, max_iter = 50000))]:
            modelo_entranado, predicciones, rmse_score, r2_score, mae_score = entrenar_regresion_lineal(X_train, y_train, X_val, y_val, modelo = modelo)
            resultados.append({'Modelo': nombre, 'Alpha': lambda_, 'RMSE': rmse_score, 'MAE': mae_score, 'R2': r2_score})
    
    return pd.DataFrame(resultados).sort_values('R2', ascending = False)