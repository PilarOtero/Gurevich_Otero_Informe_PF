import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from src.metrics import r2, rmse, mae

def entrenar_xgboost(X_train, y_train, X_val, y_val, print_ = False, n_estimators = None, max_depth = None, learning_rate = None):
    columnas_categoricas = [col for col in X_train.select_dtypes(include = ['object', 'category']).columns]
    for col in columnas_categoricas:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = pd.Categorical(X_val[col], categories=X_train[col].cat.categories)

    modelo = XGBRegressor(enable_categorical = True, random_state = 42,  n_estimators = n_estimators if n_estimators is not None else 100, 
                          max_depth = max_depth if max_depth is not None else 6, 
                          learning_rate = learning_rate if learning_rate is not None else 0.3)
    #Entrenar el arbol
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)
    
    #Métricas
    rmse_score = rmse(y_val, y_pred)
    r2_score = r2(y_val, y_pred)
    mae_score = mae(y_val, y_pred)

    if print_:
        print(f'RMSE = {rmse_score:.4f}')
        print(f'MAE = {mae_score:.4f}')
        print(f'R² = {r2_score:.4f}')

    return modelo, y_pred, round(rmse_score, 2), round(mae_score, 2), round(r2_score, 4)

def entrenar_xgboost_ohe(X_train, y_train, X_val, y_val, print_ = False,  n_estimators = None, max_depth = None, learning_rate = None):
    modelo = XGBRegressor(random_state = 42, n_estimators = n_estimators if n_estimators is not None else 100, 
                          max_depth = max_depth if max_depth is not None else 6, 
                          learning_rate = learning_rate if learning_rate is not None else 0.3)

    #Entrenar el arbol
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)

    #Métricas
    rmse_score = rmse(y_val, y_pred)
    r2_score = r2(y_val, y_pred)
    mae_score = mae(y_val, y_pred)

    if print_:
        print(f"RMSE: {rmse_score:.4f}")
        print(f'MAE = {mae_score:.4f}')
        print(f"R²:   {r2_score:.4f}")

    return modelo, y_pred, round(rmse_score, 2), round(mae_score, 2), round(r2_score, 4)

def grid_search(X_train, y_train, n_estimators_list, max_depth_list, learning_rate_list, categorico = True, folds = 5):
    kf = KFold(n_splits = folds, shuffle = True, random_state = 42)
    resultados = []

    for n_estimator in n_estimators_list:
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
                        modelo = XGBRegressor(enable_categorical = True, n_estimators = n_estimator, max_depth = max_depth, learning_rate = learning_rate, random_state = 42)
                    else:
                        modelo = XGBRegressor(n_estimators = n_estimator, max_depth = max_depth, learning_rate = learning_rate, random_state = 42)

                    modelo.fit(X_train_fold, y_train_fold)
                    y_pred = modelo.predict(X_val_fold)

                    r2_scores.append(r2(y_val_fold, y_pred))
                    rmse_scores.append(rmse(y_val_fold, y_pred))
                    mae_scores.append(mae(y_val_fold, y_pred))

                resultados.append({
                    'n_estimators': n_estimator,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'R2_mean': round(np.mean(r2_scores), 4),
                    'RMSE_mean': round(np.mean(rmse_scores), 2),
                    'MAE_mean': round(np.mean(mae_scores), 2)
                })

    return pd.DataFrame(resultados).sort_values('R2_mean', ascending = False)
