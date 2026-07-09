import zipfile

import numpy as np
import pandas as pd

from pathlib import Path
from xgboost import XGBRegressor
from IPython.display import display

from src.preprocessing import (
    corregir_marcas,
    crear_marca_modelo,
    pasar_kilometros_numerico,
    crear_0km,
    tratar_motor,
    completar_color_descripcion,
    unir_colores,
    clasificar_version,
    tratar_camara_retroceso,
)

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


#funciones para TEST
def preprocesar_test_masked(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Preprocesa el conjunto de test enmascarado (sin la columna Precio) aplicando las mismas transformaciones que en train, sin eliminar ninguna fila (los valores inválidos se reemplazan en lugar de descartarse).

        Parámetros de entrada:
            df(pd.DataFrame): dataset de test crudo, con la columna 'id' y sin la columna 'Precio'

        Parámetros de salida:
            ids(pd.Series): identificador original de cada muestra
            df(pd.DataFrame): dataset preprocesado, listo para pasar por 'preprocesamiento_post_split'
    """
    df = df.copy()
    n_inicial = len(df)

    ids = df["id"].copy()

    df = df.drop(
        columns=["id", "Unnamed: 0", "Tipo de carrocería", "Título"],
        errors="ignore"
    )

    # Mismas correcciones usadas en train
    df = corregir_marcas(df)
    df = crear_marca_modelo(df)

    # En test no eliminamos filas: valores inválidos pasan a 5
    df["Puertas"] = pd.to_numeric(df["Puertas"], errors="coerce")
    df["Puertas"] = np.where(df["Puertas"].isin([3, 5]), df["Puertas"], 5)

    df = pasar_kilometros_numerico(df)
    df = crear_0km(df)

    # En test no dropeamos Motor nulo
    df["Motor"] = df["Motor"].fillna("")
    df = tratar_motor(df)

    # Asegurar que Descripción no tenga nulos
    df["Descripción"] = df["Descripción"].fillna("")

    df = completar_color_descripcion(df)
    df = unir_colores(df)

    # En train Moneda se usó para convertir Precio; en test no hay Precio
    df = df.drop(columns=["Moneda"], errors="ignore")

    df = clasificar_version(df)
    df = tratar_camara_retroceso(df)

    if len(df) != n_inicial:
        raise ValueError(
            f"Error: se perdieron filas en test. Antes: {n_inicial}, después: {len(df)}"
        )

    return ids, df


def preparar_xgboost_categorico(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convierte a dtype 'category' las columnas categóricas del set de entrenamiento, y alinea las categorías del set de test con las de entrenamiento, para poder utilizar el soporte nativo de variables categóricas de XGBoost.

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features de entrenamiento
            X_test(pd.DataFrame): matriz de features de test

        Parámetros de salida:
            X_train(pd.DataFrame): matriz de entrenamiento con las columnas categóricas convertidas a 'category'
            X_test(pd.DataFrame): matriz de test con las mismas columnas convertidas a 'category', usando las categorías de entrenamiento
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    columnas_categoricas = X_train.select_dtypes(include=["object", "category"]).columns

    for col in columnas_categoricas:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = pd.Categorical(
            X_test[col],
            categories=X_train[col].cat.categories
        )

    return X_train, X_test


def entrenar_modelo_final_xgboost(X_full: pd.DataFrame, y_full: pd.Series, mejor_combinacion: dict) -> XGBRegressor:
    """
    Entrena el modelo XGBoost final utilizando todo el dataset etiquetado disponible (train + validación), con la mejor combinación de hiperparámetros encontrada durante la búsqueda.

        Parámetros de entrada:
            X_full(pd.DataFrame): matriz de features de todo el dataset etiquetado
            y_full(pd.Series): precio real de todo el dataset etiquetado
            mejor_combinacion(dict): combinación de hiperparámetros a utilizar (n_estimators, max_depth, learning_rate)

        Parámetros de salida:
            modelo(XGBRegressor): modelo XGBoost entrenado sobre todo el dataset
    """
    modelo = XGBRegressor(
        enable_categorical=True,
        random_state=42,
        n_estimators=int(mejor_combinacion["n_estimators"]),
        max_depth=int(mejor_combinacion["max_depth"]),
        learning_rate=float(mejor_combinacion["learning_rate"]),
        eval_metric="rmse",
    )

    modelo.fit(X_full, y_full, verbose=False)

    return modelo

def guardar_entrega_predicciones_suv(
    predicciones: np.ndarray,
    n_esperado: int = 4456,
    nombre_csv: str = "Gurevich_Otero_XGBoost_predictions.csv",
    nombre_zip: str = "Gurevich_Otero_Predictions_PF_SUV.zip"
) -> tuple[pd.DataFrame, Path, Path]:
    """
    Guarda las predicciones finales en el formato pedido por la cátedra (columnas 'id' y 'Predicted_Price_USD'), tanto en un CSV como comprimidas en un ZIP, validando la cantidad de predicciones, el shape y que no haya valores nulos.

        Parámetros de entrada:
            predicciones(np.ndarray): predicciones de precio para el conjunto de test
            n_esperado(int): cantidad de predicciones esperadas; si no coincide, se lanza un error
            nombre_csv(str): nombre (o ruta) del archivo CSV a generar
            nombre_zip(str): nombre (o ruta) del archivo ZIP a generar

        Parámetros de salida:
            df_pred(pd.DataFrame): dataframe con las columnas 'id' y 'Predicted_Price_USD' generado
            nombre_csv(Path): ruta del archivo CSV generado
            nombre_zip(Path): ruta del archivo ZIP generado
    """
    nombre_csv = Path(nombre_csv)
    nombre_zip = Path(nombre_zip)

    predicciones = np.asarray(predicciones)

    if len(predicciones) != n_esperado:
        raise ValueError(
            f"La cantidad de predicciones no coincide: "
            f"{len(predicciones)} en vez de {n_esperado}."
        )

    df_pred = pd.DataFrame({
        "id": np.arange(n_esperado),
        "Predicted_Price_USD": predicciones
    })

    if df_pred.shape != (n_esperado, 2):
        raise ValueError(f"Shape incorrecto: {df_pred.shape}")

    if df_pred["Predicted_Price_USD"].isna().sum() > 0:
        raise ValueError("Hay predicciones nulas.")

    nombre_csv.parent.mkdir(parents=True, exist_ok=True)
    nombre_zip.parent.mkdir(parents=True, exist_ok=True)

    df_pred.to_csv(nombre_csv, index=False)

    with zipfile.ZipFile(nombre_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(nombre_csv, arcname=nombre_csv.name)

    print("Entrega generada correctamente.")
    print(f"CSV: {nombre_csv}")
    print(f"ZIP: {nombre_zip}")
    print(f"Shape CSV: {df_pred.shape}")

    display(df_pred.head())
    display(df_pred.tail())

    return df_pred, nombre_csv, nombre_zip