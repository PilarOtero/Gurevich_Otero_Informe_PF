import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set_theme(style="whitegrid")
PALETA = sns.color_palette("pastel")

CELESTE = PALETA[0]
DURAZNO = PALETA[1]
VERDE = PALETA[2]
ROSA = PALETA[3]
LILA = PALETA[4]

def plot_distribuciones_por_categoria(df,categoria, variable="Log_Precio", top_n=10, bins=30):
    categorias = df[categoria].value_counts().head(top_n).index
    rows = int(np.ceil(top_n / 2))

    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    axes = np.array(axes).flatten()

    for i, cat in enumerate(categorias):
        sns.histplot(
            data=df[df[categoria] == cat],
            x=variable,
            bins=bins,
            kde=True,
            color=PALETA[i % len(PALETA)],
            edgecolor="white",
            linewidth=1,
            ax=axes[i])

        axes[i].set_title(f"{categoria}: {cat}")
        axes[i].set_xlabel(variable)
        axes[i].set_ylabel("Cantidad")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Distribuciones de {variable} según {categoria}",
        fontsize=16,
        fontweight="bold")
         
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def eda_visualizacion_suvs(df, target="Precio", current_year=2025):
    data = df.copy()

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fffafa"
    plt.rcParams["axes.edgecolor"] = "#dddddd"
    plt.rcParams["grid.alpha"] = 0.3

    # Filtro solo para visualización
    data_plot = data[
        (data["Año"] <= current_year) &
        (data["Año"] >= 1980) &
        (data[target] > 0)].copy()

    data_plot["Log_Precio"] = np.log1p(data_plot[target])
    data_plot["Log_Km"] = np.log1p(data_plot["Kilómetros"])
    data_plot["Antiguedad"] = (current_year - data_plot["Año"]).clip(lower=0)

    # ==================================================
    # 1. DISTRIBUCIONES PRINCIPALES
    # ==================================================

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    sns.histplot(data_plot[target], bins=40, kde=True, color=ROSA, edgecolor="white", ax=axes[0])
    axes[0].set_title("Distribución de Precio")
    axes[0].set_xlabel("Precio")
    axes[0].set_ylabel("Cantidad")

    sns.histplot(data_plot["Log_Precio"], bins=40, kde=True, color=LILA, edgecolor="white", ax=axes[1])
    axes[1].set_title("Distribución de Log(Precio)")
    axes[1].set_xlabel("Log(Precio)")
    axes[1].set_ylabel("Cantidad")

    sns.histplot(data_plot["Kilómetros"].dropna(), bins=40, kde=True, color=CELESTE, edgecolor="white", ax=axes[2])
    axes[2].set_title("Distribución de Kilómetros")
    axes[2].set_xlabel("Kilómetros")
    axes[2].set_ylabel("Cantidad")

    sns.histplot(data_plot["Log_Km"].dropna(), bins=40, kde=True, color=DURAZNO, edgecolor="white", ax=axes[3])
    axes[3].set_title("Distribución de Log(Kilómetros)")
    axes[3].set_xlabel("Log(Kilómetros)")
    axes[3].set_ylabel("Cantidad")

    plt.suptitle("Distribuciones principales", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ==================================================
    # 2. DISTRIBUCIONES POR MARCA_MODELO
    # ==================================================

    if "Marca_Modelo" in data_plot.columns:
        plot_distribuciones_por_categoria(
            data_plot,
            categoria = "Marca_Modelo",
            variable = "Log_Precio",
            top_n = 6)
   # ==================================================
    # 10. PRECIO PROMEDIO POR MARCA_MODELO
    # ==================================================

    if "Marca_Modelo" in data_plot.columns:
        top_modelos = (
            data_plot["Marca_Modelo"]
            .value_counts()
            .head(12)
            .index)

        precio_medio_modelo = (
            data_plot[data_plot["Marca_Modelo"].isin(top_modelos)]
            .groupby("Marca_Modelo")[target]
            .mean()
            .sort_values(ascending=False))

        plt.figure(figsize=(12, 6))

        sns.barplot(
            x=precio_medio_modelo.values,
            y=precio_medio_modelo.index,
            color= LILA)

        plt.title("Precio promedio por Marca_Modelo más frecuente", fontsize=14, fontweight="bold")
        plt.xlabel("Precio promedio")
        plt.ylabel("Marca_Modelo")
        plt.tight_layout()
        plt.show()
    # ==================================================
    # 3. DISTRIBUCIONES POR TIPO DE COMBUSTIBLE
    # ==================================================

    if "Tipo de combustible" in data_plot.columns:
        plot_distribuciones_por_categoria(
            data_plot,
            categoria = "Tipo de combustible",
            variable = "Log_Precio",
            top_n = 6)


    # ==================================================
    # BOXPLOTS CATEGÓRICOS RESPECTO A PRECIO
    # ==================================================

    _, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # 1) Transmisión
    orden_trans = (
        data_plot
        .groupby("Transmisión")["Log_Precio"]
        .median()
        .sort_values()
        .index)

    sns.boxplot(
        data=data_plot,
        x="Transmisión",
        y="Log_Precio",
        order=orden_trans,
        color=ROSA,
        ax=axes[0])

    axes[0].set_title("Según transmisión")
    axes[0].set_xlabel("Transmisión")
    axes[0].set_ylabel("Log(Precio)")
    axes[0].tick_params(axis="x", rotation=35)

    # 2) Tipo de vendedor
    orden_vendedor = (
        data_plot
        .groupby("Tipo de vendedor")["Log_Precio"]
        .median()
        .sort_values()
        .index
    )
    sns.boxplot(
        data=data_plot,
        x="Tipo de vendedor",
        y="Log_Precio",
        order=orden_vendedor,
        color=CELESTE,
        ax=axes[1])

    axes[1].set_title("Según tipo de vendedor")
    axes[1].set_xlabel("Tipo de vendedor")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=25)

    # 3) Condición 0km
    data_plot["Condición"] = data_plot["0km"].map({
        0: "Usado",
        1: "0km"})

    sns.boxplot(
        data=data_plot,
        x="Condición",
        y="Log_Precio",
        order=["Usado", "0km"],
        color=LILA,
        ax=axes[2])

    axes[2].set_title("Según condición")
    axes[2].set_xlabel("Condición")
    axes[2].set_ylabel("")

    plt.suptitle(
        "Distribución de Log(Precio) según variables categóricas",
        fontsize=16,
        fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    # ==================================================
    # 11. MATRIZ DE CORRELACIÓN
    # ==================================================

    corr_cols = [
        target,
        "Log_Precio",
        "Kilómetros",
        "Log_Km",
        "Antiguedad",
        "Puertas",
        "0km",
        "Motor_Litros",
        "Score Descripción"]

    corr_cols = [col for col in corr_cols if col in data_plot.columns]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        data_plot[corr_cols].corr(),
        annot=True,
        cmap="mako",
        fmt=".2f",
        linewidths=0.5,
        square=True)

    plt.title("Matriz de correlación", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_precio_segun_antiguedad_km(df, target="Precio", current_year=2025 ):
    data = df.copy()

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fffafa"
    plt.rcParams["axes.edgecolor"] = "#dddddd"
    plt.rcParams["grid.alpha"] = 0.3

    # Filtro solo para visualización
    data_plot = data[
        (data["Año"] <= current_year) &
        (data["Año"] >= 1980) &
        (data[target] > 0)].copy()

    data_plot["Log_Precio"] = np.log1p(data_plot[target])
    data_plot["Log_Km"] = np.log1p(data_plot["Kilómetros"])
    data_plot["Antiguedad"] = (current_year - data_plot["Año"]).clip(lower=0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Kilómetros vs Precio
    sns.regplot(
        data=data_plot.dropna(subset=["Log_Km", "Log_Precio"]),
        x="Log_Km",
        y="Log_Precio",
        color=CELESTE,
        scatter_kws={"alpha": 0.15, "s": 12},
        line_kws={"linewidth": 2},
        ax=axes[0])

    axes[0].set_title("Kilometraje vs Precio")
    axes[0].set_xlabel("Log(Kilómetros)")
    axes[0].set_ylabel("Log(Precio)")

    # Antigüedad vs Precio
    data_ant = data_plot[data_plot["Antiguedad"] <= 50].copy()

    sns.regplot(
        data=data_ant,
        x="Antiguedad",
        y="Log_Precio",
        color=LILA,
        scatter_kws={"alpha": 0.15, "s": 12},
        line_kws={"linewidth": 2},
        ax=axes[1])

    axes[1].set_title("Antigüedad vs Precio")
    axes[1].set_xlabel("Antigüedad (años)")
    axes[1].set_ylabel("Log(Precio)")

    plt.suptitle(
        "Relación del precio con kilometraje y antigüedad",
        fontsize=16,
        fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_precio_segun_rango_ant(df, target="Precio"):
    data_plot = df[df["Antiguedad"] <= 50].copy()

    data_plot["Rango antigüedad"] = pd.cut(
        data_plot["Antiguedad"],
        bins=[0, 2, 5, 10, 15, 20, 100],
        labels=["0-2", "3-5", "6-10", "11-15", "16-20", "20+"])
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=data_plot,
        x="Rango antigüedad",
        y=target)
    
    plt.title("Precio según rango de antigüedad", fontsize=14, fontweight="bold")
    plt.xlabel("Antigüedad (años)")
    plt.ylabel("Precio")
    plt.tight_layout()
    plt.show()

def plot_dispersion_por_marca(df, target="Precio", min_muestras=35, top_n=15):
    data = df.copy()

    marcas_validas = data["Marca"].value_counts()
    marcas_validas = marcas_validas[marcas_validas >= min_muestras].index

    data_filtrado = data[data["Marca"].isin(marcas_validas)]

    cv = (
        data_filtrado
        .groupby("Marca")[target]
        .agg(["count", "mean", "std"]))

    cv["cv"] = cv["std"] / cv["mean"]
    cv = cv.dropna()

    # Más dispersas y menos dispersas
    cv_alto = cv.sort_values("cv", ascending = False).head(top_n)
    cv_bajo = cv.sort_values("cv", ascending = True).head(top_n).sort_values("cv", ascending = False)

    _, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.barplot(
        data=cv_alto.reset_index(),
        x = "cv",
        y = "Marca",
        ax=axes[0])
    
    axes[0].set_title(f"Top {top_n} modelos con mayor dispersión")
    axes[0].set_xlabel("Coeficiente de variación")
    axes[0].set_ylabel("Marca")

    sns.barplot(
        data=cv_bajo.reset_index(),
        x = "cv",
        y = "Marca",
        ax=axes[1])

    axes[1].set_title(f"Top {top_n} modelos con menor dispersión")
    axes[1].set_xlabel("Coeficiente de variación")
    axes[1].set_ylabel("")

    plt.suptitle("Dispersión relativa de precios por Marca", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Boxplot solo de los de menor dispersión
    orden = cv_bajo.index.tolist()

    plt.figure(figsize=(16, 6))
    sns.boxplot(
        data=data_filtrado[data_filtrado["Marca"].isin(orden)],
        x = "Marca",
        y = target,
        order = orden)
    
    plt.yscale('log')
    plt.title(f"Distribución de precios - {top_n} modelos más uniformes")
    plt.xlabel("Marca")
    plt.ylabel("log(Precio)")
    plt.xticks(rotation = 45, ha = "right")
    plt.tight_layout()
    plt.show()

    return cv.sort_values("cv")


#PARA OUTLIERS
def plot_boxplots_por_grupo(
    dataset: pd.DataFrame,
    grupo_col: str = "Marca_Modelo",
    col: str = "Precio",
    top_n: int = 12
) -> None:

    top_grupos = dataset[grupo_col].value_counts().head(top_n).index
    df_top = dataset[dataset[grupo_col].isin(top_grupos)].copy()

    orden = (
        df_top.groupby(grupo_col)[col]
        .median()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df_top,
        x=grupo_col,
        y=col,
        order=orden,
        color="lightsteelblue"
    )

    plt.xticks(rotation=40, ha="right")
    plt.title(f"Distribución de {col} por {grupo_col}")
    plt.xlabel(grupo_col)
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


def plot_grupo_especifico(dataset: pd.DataFrame, grupo: str, grupo_col: str = "Marca_Modelo", col: str = "Precio", k: float = 1.5) -> pd.DataFrame:
    data_grupo = dataset[dataset[grupo_col] == grupo].copy()
    valores = data_grupo[col].dropna()

    if len(valores) == 0:
        print(f"No hay registros para {grupo}.")
        return pd.DataFrame()

    q1 = valores.quantile(0.25)
    q3 = valores.quantile(0.75)
    iqr = q3 - q1

    limite_inf = q1 - k * iqr
    limite_sup = q3 + k * iqr

    outliers = data_grupo[
        (data_grupo[col] < limite_inf) |
        (data_grupo[col] > limite_sup)
    ]

    print(f"{grupo} — {len(data_grupo)} registros")
    print(f"Q1={q1:,.2f}")
    print(f"Q3={q3:,.2f}")
    print(f"IQR={iqr:,.2f}")
    print(f"Rango válido: [{limite_inf:,.2f}, {limite_sup:,.2f}]")
    print(f"Outliers detectados: {len(outliers)}")

    #columnas = [
    #    c for c in [
    #        "Marca_Modelo",
    #        "Precio",
    #        "Año",
    #        "Kilómetros",
    #        "Motor_Litros",
    #        "0km"
    #    ]
    #    if c in data_grupo.columns]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(valores, bins=30, kde=True, color="lightsteelblue", ax=axes[0])
    axes[0].axvline(limite_inf, color="crimson", linestyle="--", label="Límite inferior")
    axes[0].axvline(limite_sup, color="crimson", linestyle="--", label="Límite superior")
    axes[0].set_title(f"Distribución de {col} para {grupo}")
    axes[0].set_xlabel(col)
    axes[0].legend()

    sns.boxplot(x=valores, color="lightsteelblue", ax=axes[1])
    axes[1].axvline(limite_inf, color="crimson", linestyle="--", label="Límite inferior")
    axes[1].axvline(limite_sup, color="crimson", linestyle="--", label="Límite superior")
    axes[1].set_title(f"Boxplot de {col} para {grupo}")
    axes[1].set_xlabel(col)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return outliers