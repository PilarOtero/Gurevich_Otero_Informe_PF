import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def eda_visualizacion_suvs(
    df,
    target="Precio",
    features_num=None,
    show_pairplot=False
):
    """
    Visualizaciones EDA para el dataset de SUVs.
    Incluye distribuciones, boxplots, relaciones con Precio y correlación.
    """

    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fafafa"
    plt.rcParams["axes.edgecolor"] = "#cccccc"

    if features_num is None:
        features_num = [
            "Precio",
            "Año",
            "Puertas",
            "Kilómetros",
            "0km",
            "Antiguedad",
            "Km_por_anio",
            "Log_Km"
        ]

    features_num = [col for col in features_num if col in df.columns]

    print("Distribuciones individuales")

    rows = int(np.ceil(len(features_num) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 5 * rows))
    axes = axes.flatten()

    colores = sns.color_palette("Set2", len(features_num))

    for i, col in enumerate(features_num):
        sns.histplot(
            df[col].dropna(),
            kde=True,
            bins=30,
            ax=axes[i],
            color=colores[i],
            edgecolor="white",
            alpha=0.9
        )
        axes[i].set_title(f"Distribución de {col}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Distribuciones de variables numéricas", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("Boxplots individuales")

    rows = int(np.ceil(len(features_num) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(features_num):
        sns.boxplot(
            y=df[col],
            ax=axes[i],
            color=colores[i],
            width=0.4
        )
        axes[i].set_title(f"Boxplot de {col}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Detección de outliers", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("Relación con el precio")

    features_scatter = [col for col in features_num if col != target]

    rows = int(np.ceil(len(features_scatter) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(features_scatter):
        sns.scatterplot(
            data=df,
            x=col,
            y=target,
            ax=axes[i],
            alpha=0.35,
            s=25
        )
        axes[i].set_title(f"{target} vs {col}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Relación entre variables numéricas y precio", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("Relación de variables categóricas con el precio")

    categorical_plots = [
        "Marca",
        "Modelo",
        "Color",
        "Tipo de combustible",
        "Transmisión",
        "Tipo de vendedor"
    ]

    for col in categorical_plots:
        if col in df.columns:
            top_categories = df[col].value_counts().head(15).index

            plt.figure(figsize=(14, 6))
            sns.boxplot(
                data=df[df[col].isin(top_categories)],
                x=col,
                y=target
            )
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{target} según {col}")
            plt.tight_layout()
            plt.show()

    if show_pairplot:
        print("Pairplot")
        sns.pairplot(
            df[features_num].dropna(),
            corner=True,
            diag_kind="hist",
            plot_kws={"alpha": 0.4, "s": 20}
        )
        plt.suptitle("Relaciones bivariadas", y=1.02, fontsize=16, fontweight="bold")
        plt.show()

    print("Matriz de correlación")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df[features_num].corr(),
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        linewidths=0.5,
        square=True
    )
    plt.title("Matriz de correlación", fontsize=16, fontweight="bold")
    plt.show()