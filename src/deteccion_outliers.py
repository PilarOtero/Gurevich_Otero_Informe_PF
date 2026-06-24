import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def flagear_outliers_por_grupo(
    dataset: pd.DataFrame,
    grupo_col: str = "Marca_Modelo",
    col: str = "Precio",
    k: float = 1.5,
    min_registros: int = 20
) -> pd.DataFrame:

    dataset = dataset.copy()
    flag_col = f"outlier_{col.lower().replace(' ', '_')}_grupo"
    dataset[flag_col] = 0

    grupos_chicos = []

    for grupo, idx in dataset.groupby(grupo_col).groups.items():

        if len(idx) < min_registros:
            grupos_chicos.append(grupo)
            continue

        valores = dataset.loc[idx, col].dropna()

        q1 = valores.quantile(0.25)
        q3 = valores.quantile(0.75)
        iqr = q3 - q1

        limite_inf = q1 - k * iqr
        limite_sup = q3 + k * iqr

        mascara_out = (
            (dataset.loc[idx, col] < limite_inf) |
            (dataset.loc[idx, col] > limite_sup)
        )

        dataset.loc[idx[mascara_out], flag_col] = 1

    n_out = dataset[flag_col].sum()

    print(f"[{col}] Outliers detectados: {n_out} ({100 * n_out / len(dataset):.2f}%)")
    print(f"Grupos chicos no filtrados (<{min_registros} registros): {len(grupos_chicos)}")

    return dataset

def reportar_outliers_por_grupo(dataset: pd.DataFrame, grupo_col: str = "Marca_Modelo", col: str = "Precio", top_n: int = 20) -> pd.DataFrame:
    flag_col = f"outlier_{col.lower().replace(' ', '_')}_grupo"

    resumen = (dataset[dataset[flag_col] == 1].groupby(grupo_col).agg(
            n_outliers=(flag_col, "sum"),
            precio_min=(col, "min"),
            precio_max=(col, "max"),
            precio_mediana=(col, "median")
        )
        .sort_values("n_outliers", ascending=False).head(top_n))

    #display(resumen.round(2))
    return resumen

def ver_outliers(dataset: pd.DataFrame, col: str = "Precio", n: int = 50) -> pd.DataFrame:
    flag_col = f"outlier_{col.lower().replace(' ', '_')}_grupo"

    columnas = [
        c for c in [
            "Marca_Modelo",
            "Precio",
            "Año",
            "Kilómetros",
        ] if c in dataset.columns]

    outliers = (
        dataset[dataset[flag_col] == 1].sort_values(col, ascending=False)[columnas].head(n))

    display(outliers)
    #return outliers

def eliminar_outliers_grupo(dataset: pd.DataFrame, col: str = "Precio") -> pd.DataFrame:
    flag_col = f"outlier_{col.lower().replace(' ', '_')}_grupo"

    if flag_col not in dataset.columns:
        raise ValueError(f"No existe la columna {flag_col}. Primero tenés que flagear outliers.")

    n_antes = len(dataset)

    dataset_limpio = (dataset[dataset[flag_col] == 0].drop(columns=[flag_col]).copy())
    n_despues = len(dataset_limpio)

    print(f"Registros antes: {n_antes}")
    print(f"Registros después: {n_despues}")
    print(f"Eliminados: {n_antes - n_despues} ({100 * (n_antes - n_despues) / n_antes:.2f}%)")

    return dataset_limpio


def eliminar_outliers_por_corte(dataset: pd.DataFrame, precio_min: float = 5000) -> pd.DataFrame:
    dataset = dataset.copy()

    n_antes = len(dataset)

    dataset_limpio = dataset[(dataset["Precio"] > precio_min)].copy()
    n_despues = len(dataset_limpio)
    print(f"Eliminados: {n_antes - n_despues} ({100 * (n_antes - n_despues) / n_antes:.2f}%)")

    return dataset_limpio