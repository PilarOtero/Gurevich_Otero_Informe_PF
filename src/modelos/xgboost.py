import pandas as pd
from xgboost import XGBRegressor
from src.metrics import r2, rmse, mae

def entrenar_xgboost(X_train, y_train, X_val, y_val):
    columnas_categoricas = [col for col in X_train.select_dtypes(include = ['object', 'category']).columns]
    for col in columnas_categoricas:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = pd.Categorical(X_val[col], categories=X_train[col].cat.categories)

    modelo = XGBRegressor(enable_categorical = True, random_state = 42)
    #Entrenar el arbol
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_val)
    rmse_score = rmse(y_val, y_pred)
    r2_score = r2(y_val, y_pred)
    mae_score = mae(y_val, y_pred)

    #Métricas
    print(f'RMSE = {rmse_score:.4f}')
    print(f'MAE = {mae_score:.4f}')
    print(f'R² = {r2_score:.4f}')

    return modelo, y_pred

def entrenar_xgboost_ohe(X_train, y_train, X_val, y_val):
    modelo = XGBRegressor(random_state = 42)

    #Entrenar el arbol
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)

    #Métricas
    rmse_score = rmse(y_val, y_pred)
    r2_val = r2(y_val, y_pred)
    mae_score = mae(y_val, y_pred)

    print(f"RMSE: {rmse_score:.4f}")
    print(f'MAE = {mae_score:.4f}')
    print(f"R²:   {r2_val:.4f}")

    return modelo, y_pred
