import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distribuciones_por_categoria(df, categoria, variable = "Log_Precio", top_n = 6, bins = 30):
    categorias = df[categoria].value_counts().head(top_n).index
    rows = int(np.ceil(top_n / 3))

    _, axes = plt.subplots(rows, 3, figsize=(16, 5 * rows))
    axes = axes.flatten()

    for i, cat in enumerate(categorias):
        sns.histplot(
            data=df[df[categoria] == cat],
            x=variable,
            bins=bins,
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(f"{categoria}: {cat}")
        axes[i].set_xlabel(variable)
        axes[i].set_ylabel("Cantidad")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Distribuciones de {variable} según {categoria}",
        fontsize=16,
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def eda_visualizacion_suvs(df, target = "Precio", current_year = 2024):
    data = df.copy()

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # Filtro solo para visualización
    data_plot = data[
        (data["Año"] <= current_year) &
        (data["Año"] >= 1980)
    ].copy()

    data_plot["Log_Precio"] = np.log1p(data_plot[target])
    data_plot["Log_Km"] = np.log1p(data_plot["Kilómetros"])
    data_plot["Antiguedad"] = current_year - data_plot["Año"]

    # ==============================
    # 1. DISTRIBUCIONES PRINCIPALES
    # ==============================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    sns.histplot(data_plot[target], bins=40, kde=True, ax=axes[0])
    axes[0].set_title("Distribución de Precio")
    axes[0].set_xlabel("Precio")

    sns.histplot(data_plot["Log_Precio"], bins=40, kde=True, ax=axes[1])
    axes[1].set_title("Distribución de Log(Precio)")
    axes[1].set_xlabel("Log(Precio)")

    sns.histplot(data_plot["Kilómetros"].dropna(), bins=40, kde=True, ax=axes[2])
    axes[2].set_title("Distribución de Kilómetros")
    axes[2].set_xlabel("Kilómetros")

    sns.histplot(data_plot["Log_Km"].dropna(), bins=40, kde=True, ax=axes[3])
    axes[3].set_title("Distribución de Log(Kilómetros)")
    axes[3].set_xlabel("Log(Kilómetros)")

    plt.suptitle("Distribuciones principales", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ==============================
    # 2. DISTRIBUCIONES POR CATEGORIA
    # ==============================

    plot_distribuciones_por_categoria(
        data_plot,
        categoria="Marca",
        variable="Log_Precio",
        top_n=6
    )

    plot_distribuciones_por_categoria(
        data_plot,
        categoria="Tipo de combustible",
        variable="Log_Precio",
        top_n=6
    )

    # ==============================
    # 3. RELACION PRECIO - KILOMETROS
    # ==============================

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=data_plot,
        x="Log_Km",
        y="Log_Precio",
        alpha=0.25,
        s=18
    )

    plt.title("Relación entre precio y kilometraje")
    plt.xlabel("Log(Kilómetros)")
    plt.ylabel("Log(Precio)")
    plt.tight_layout()
    plt.show()

    # ==============================
    # 4. BOXPLOTS INDIVIDUALES
    # ==============================

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    sns.boxplot(
        data=data_plot,
        x="Transmisión",
        y=target,
        ax=axes[0]
    )
    axes[0].set_title("Precio según transmisión")
    axes[0].set_xlabel("Transmisión")
    axes[0].set_ylabel("Precio")
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(
        data=data_plot,
        x="Tipo de vendedor",
        y=target,
        ax=axes[1]
    )
    axes[1].set_title("Precio según tipo de vendedor")
    axes[1].set_xlabel("Tipo de vendedor")
    axes[1].set_ylabel("Precio")
    axes[1].tick_params(axis="x", rotation=45)

    if "0km" in data_plot.columns:
        data_plot["0km_label"] = data_plot["0km"].map({
            0: "Usado",
            1: "0km"
        })

        sns.boxplot(
            data=data_plot,
            x="0km_label",
            y=target,
            ax=axes[2]
        )
        axes[2].set_title("Precio según condición")
        axes[2].set_xlabel("Condición")
        axes[2].set_ylabel("Precio")
    else:
        axes[2].axis("off")

    plt.suptitle("Boxplots individuales", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ==============================
    # 5. MATRIZ DE CORRELACION
    # ==============================

    corr_cols = [
        target,
        "Log_Precio",
        "Kilómetros",
        "Log_Km",
        "Año",
        "Antiguedad",
        "Puertas",
        "0km"
    ]

    corr_cols = [col for col in corr_cols if col in data_plot.columns]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        data_plot[corr_cols].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()

def plot_antiguedad_km_vs_precio(df, target="Precio"):
    data_plot = data_plot = df[df["Antiguedad"] <= 50].copy()
    data_plot["Log_Precio"] = np.log1p(data_plot[target])
    data_plot["Log_Km"] = np.log1p(data_plot["Kilómetros"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Antiguedad vs Log_Precio
    sns.scatterplot(
        data=data_plot,
        x="Antiguedad",
        y="Log_Precio",
        alpha=0.25,
        s=18,
        ax=axes[0]
    )
    axes[0].set_title("Antigüedad vs Precio")
    axes[0].set_xlabel("Antigüedad (años)")
    axes[0].set_ylabel("Log(Precio)")

    # Boxplot por rangos de antigüedad
    data_plot["Rango antigüedad"] = pd.cut(
        data_plot["Antiguedad"],
        bins=[0, 2, 5, 10, 15, 20, 100],
        labels=["0-2", "3-5", "6-10", "11-15", "16-20", "20+"]
    )
    sns.boxplot(
        data=data_plot,
        x="Rango antigüedad",
        y=target,
        ax=axes[1]
    )
    axes[1].set_title("Precio según rango de antigüedad")
    axes[1].set_xlabel("Antigüedad (años)")
    axes[1].set_ylabel("Precio")

    plt.suptitle("Efecto de antigüedad y kilometraje sobre el precio", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_dispersion_por_marca(df, target="Precio", top_n=15):
    marcas_validas = df['Marca'].value_counts()
    marcas_validas = marcas_validas[marcas_validas >= 35].index
    data_filtrado = df[df['Marca'].isin(marcas_validas)]

    cv = data_filtrado.groupby('Marca')['Precio'].agg(['std', 'mean'])
    cv["cv"] = cv["std"] / cv["mean"]
    cv = cv.sort_values("cv", ascending = False)

    _, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Coeficiente de variación
    sns.barplot(x = cv["cv"], y = cv.index, ax = axes[0])
    axes[0].set_title("Coeficiente de variación por marca (menor = más uniforme)")
    axes[0].set_xlabel("CV (std / mean)")
    axes[0].set_ylabel("Marca")

    # Boxplot de las marcas con menor dispersión
    marcas_low_cv = cv.index.tolist()
    sns.boxplot(
        data=df[df["Marca"].isin(marcas_low_cv)],
        x = "Marca",
        y=target,
        order = cv.index,
        ax=axes[1]
    )
    axes[1].set_title("Distribución de precios (marcas con menor dispersión)")
    axes[1].set_xlabel("Marca")
    axes[1].set_ylabel("Precio")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("Dispersión de precios por marca", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
