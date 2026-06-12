import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#plot del 2.1


def eda_visualizacion(
    df,
    features_num=None,
    target="precio",
    show_pairplot=True
):

    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fafafa"
    plt.rcParams["axes.edgecolor"] = "#cccccc"

    if features_num is None:
        features_num = ["precio", "Área", "metros_cubiertos", "ambientes", "pisos", "edad"]

    colores = sns.color_palette("Set2", len(features_num))

    
    print("Distribuciones individuales")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(features_num):
        sns.histplot(
            df[col],
            kde=True,
            bins=30,
            ax=axes[i],
            color=colores[i],
            edgecolor="white",
            alpha=0.9
        )
        axes[i].set_title(f"Distribución de {col}")

    plt.suptitle("Distribuciones de variables", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("Boxplots individuales")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(features_num):
        sns.boxplot(
            y=df[col],
            ax=axes[i],
            color=colores[i],
            width=0.4
        )
        axes[i].set_title(f"Boxplot de {col}")

    plt.suptitle("Detección de outliers", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[features_num], palette="Set2")
    plt.title("Boxplots combinados", fontsize=16, fontweight="bold")
    plt.xticks(rotation=20)
    plt.show()

   
    print("Relación con el precio")

    features_scatter = [col for col in features_num if col != target]

    n = len(features_scatter)
    rows = int(np.ceil(n / 3))

    fig, axes = plt.subplots(rows, 3, figsize=(16, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(features_scatter):
        sns.scatterplot(
            x=col,
            y=target,
            data=df,
            ax=axes[i],
            alpha=0.6,
            s=35
        )
        axes[i].set_title(f"{target} vs {col}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Relación entre variables y precio", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


    if "tipo" in df.columns:
        plt.figure(figsize=(9, 6))
        sns.boxplot(
            x="tipo",
            y=target,
            hue="tipo",            
            data=df,
            palette="Pastel1",
            legend=False           
        )
        plt.title("Precio según tipo de propiedad", fontsize=16, fontweight="bold")
        plt.xlabel("Tipo de propiedad")
        plt.ylabel("Precio")
        plt.show()


    if show_pairplot:
        print("Pairplot")
        sns.pairplot(
            df[features_num],
            corner=True,
            diag_kind="hist",
            plot_kws={"alpha": 0.5, "s": 20}
        )
        plt.suptitle("Relaciones bivariadas", y=1.02, fontsize=16, fontweight="bold")
        plt.show()


    plt.figure(figsize=(9, 7))
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
