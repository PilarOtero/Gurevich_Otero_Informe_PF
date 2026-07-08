import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from src.metrics import r2, rmse, mae

def entrenar_xgboost(X_train:pd.DataFrame, y_train:pd.Series, X_val:pd.DataFrame, y_val:pd.Series, print_:bool = False, n_estimators:int = None, max_depth:int = None, learning_rate:float = None) -> tuple[XGBRegressor, np.ndarray, float, float, float, pd.DataFrame]:
    """
    Entrena un modelo XGBoost utilizando el soporte nativo de variables categóricas (enable_categorical = True), convirtiendo a tipo 'category' las columnas object/category del set de entrenamiento y aplicando las mismas categorías al set de validación.

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features de entrenamiento
            y_train(pd.Series): precio real del set de entrenamiento
            X_val(pd.DataFrame): matriz de features de validación
            y_val(pd.Series): precio real del set de validación
            print_(bool): si es True, imprime las métricas de validación por pantalla
            n_estimators(int): cantidad de árboles del modelo (si es None, se usa el valor por defecto 100)
            max_depth(int): profundidad máxima de cada árbol (si es None, se usa el valor por defecto 6)
            learning_rate(float): tasa de aprendizaje del modelo (si es None, se usa el valor por defecto 0.3)

        Parámetros de salida:
            modelo(XGBRegressor): modelo entrenado
            y_pred(np.ndarray): predicciones del modelo sobre el set de validación
            rmse_score(float): RMSE sobre el set de validación, redondeado
            mae_score(float): MAE sobre el set de validación, redondeado
            r2_score(float): R² sobre el set de validación, redondeado
            historial(pd.DataFrame): RMSE de train y de validación por cada árbol entrenado
    """
    columnas_categoricas = [col for col in X_train.select_dtypes(include = ['object', 'category']).columns]
    for col in columnas_categoricas:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = pd.Categorical(X_val[col], categories=X_train[col].cat.categories)

    modelo = XGBRegressor(enable_categorical = True, random_state = 42,  n_estimators = n_estimators if n_estimators is not None else 100, 
                          max_depth = max_depth if max_depth is not None else 6, 
                          learning_rate = learning_rate if learning_rate is not None else 0.3,
                          eval_metric = 'rmse')
    #Entrenar el arbol
    modelo.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], verbose = False)
    evals = modelo.evals_result()
    historial = pd.DataFrame({
        'Arbol': range(len(evals['validation_0']['rmse'])),
        'train_rmse': evals['validation_0']['rmse'],
        'val_rmse': evals['validation_1']['rmse']
    })
    
    y_pred = modelo.predict(X_val)
    
    #Métricas
    rmse_score = rmse(y_val, y_pred)
    r2_score = r2(y_val, y_pred)
    mae_score = mae(y_val, y_pred)

    if print_:
        print(f'RMSE = {rmse_score:.4f}')
        print(f'MAE = {mae_score:.4f}')
        print(f'R² = {r2_score:.4f}')

    return modelo, y_pred, round(rmse_score, 2), round(mae_score, 2), round(r2_score, 4), historial

def entrenar_xgboost_ohe(X_train:pd.DataFrame, y_train:pd.Series, X_val:pd.DataFrame, y_val:pd.Series, print_:bool = False, n_estimators:int = None, max_depth:int = None, learning_rate:float = None):
    """
    Entrena un modelo XGBoost sobre datos previamente codificados con One-Hot Encoding (sin utilizar el soporte nativo de variables categóricas de XGBoost).

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features de entrenamiento, con variables categóricas ya codificadas (OHE)
            y_train(pd.Series): precio real del set de entrenamiento
            X_val(pd.DataFrame): matriz de features de validación, con variables categóricas ya codificadas (OHE)
            y_val(pd.Series): precio real del set de validación
            print_(bool): si es True, imprime las métricas de validación por pantalla
            n_estimators(int): cantidad de árboles del modelo (si es None, se usa el valor por defecto 100)
            max_depth(int): profundidad máxima de cada árbol (si es None, se usa el valor por defecto 6)
            learning_rate(float): tasa de aprendizaje del modelo (si es None, se usa el valor por defecto 0.3)

        Parámetros de salida:
            modelo(XGBRegressor): modelo entrenado
            y_pred(np.ndarray): predicciones del modelo sobre el set de validación
            rmse_score(float): RMSE sobre el set de validación, redondeado
            mae_score(float): MAE sobre el set de validación, redondeado
            r2_score(float): R² sobre el set de validación, redondeado
            historial(pd.DataFrame): RMSE de train y de validación por cada árbol entrenado
    """
    modelo = XGBRegressor(random_state = 42, n_estimators = n_estimators if n_estimators is not None else 100, 
                          max_depth = max_depth if max_depth is not None else 6, 
                          learning_rate = learning_rate if learning_rate is not None else 0.3,
                          eval_metric = 'rmse')

    #Entrenar el arbol
    modelo.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], verbose = False)
    evals = modelo.evals_result()
    historial = pd.DataFrame({
        'Arbol': range(len(evals['validation_0']['rmse'])),
        'train_rmse': evals['validation_0']['rmse'],
        'val_rmse': evals['validation_1']['rmse']
    })

    y_pred = modelo.predict(X_val)

    #Métricas
    rmse_score = rmse(y_val, y_pred)
    r2_score = r2(y_val, y_pred)
    mae_score = mae(y_val, y_pred)

    if print_:
        print(f"RMSE: {rmse_score:.4f}")
        print(f'MAE = {mae_score:.4f}')
        print(f"R²:   {r2_score:.4f}")

    return modelo, y_pred, round(rmse_score, 2), round(mae_score, 2), round(r2_score, 4), historial

def grid_search(X_train:pd.DataFrame, y_train:pd.Series, n_estimators_list:list, max_depth_list:list, learning_rate_list:list, categorico:bool = True, folds:int = 5):
    """
    Busca la mejor combinación de hiperparámetros (n_estimators, max_depth, learning_rate) para XGBoost mediante validación cruzada K-Fold, evaluando todas las combinaciones posibles entre las listas recibidas.

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features de entrenamiento
            y_train(pd.Series): precio real del set de entrenamiento
            n_estimators_list(list): valores de cantidad de árboles a probar
            max_depth_list(list): valores de profundidad máxima a probar
            learning_rate_list(list): valores de tasa de aprendizaje a probar
            categorico(bool): si es True, trata las columnas object/category como categóricas nativas de XGBoost; si es False, asume que las variables categóricas ya vienen codificadas (ej. con One-Hot Encoding)
            folds(int): cantidad de folds utilizados en la validación cruzada

        Parámetros de salida:
            resultados(pd.DataFrame): promedio de R², RMSE y MAE por cada combinación de hiperparámetros, ordenado de menor a mayor RMSE promedio
    """
    kf = KFold(n_splits = folds, shuffle = True, random_state = 42)
    resultados = []

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
                r2_scores, rmse_scores, mae_scores = [], [], []
                
                for train_idx, val_idx in kf.split(X_train):
                    X_train_fold = X_train.iloc[train_idx].copy()
                    X_val_fold = X_train.iloc[val_idx].copy()
                    y_train_fold = y_train.iloc[train_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    if categorico:
                        for col in X_train_fold.select_dtypes(include = ['object', 'category']).columns:
                            X_train_fold[col] = X_train_fold[col].astype('category')
                            X_val_fold[col] = pd.Categorical(X_val_fold[col], categories = X_train_fold[col].cat.categories)
                        modelo = XGBRegressor(enable_categorical = True, n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, random_state = 42)
                    else:
                        modelo = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, random_state = 42)

                    modelo.fit(X_train_fold, y_train_fold)
                    y_pred = modelo.predict(X_val_fold)

                    r2_scores.append(r2(y_val_fold, y_pred))
                    rmse_scores.append(rmse(y_val_fold, y_pred))
                    mae_scores.append(mae(y_val_fold, y_pred))

                resultados.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'R2_mean': round(np.mean(r2_scores), 4),
                    'RMSE_mean': round(np.mean(rmse_scores), 2),
                    'MAE_mean': round(np.mean(mae_scores), 2)
                })

    return pd.DataFrame(resultados).sort_values('RMSE_mean', ascending = True).reset_index(drop = True)
