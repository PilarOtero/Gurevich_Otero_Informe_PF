import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribuciones_por_categoria(
    df,
    categoria,
    variable="Log_Precio",
    top_n=6,
    bins=30
):
    categorias = df[categoria].value_counts().head(top_n).index
    rows = int(np.ceil(top_n / 3))

    fig, axes = plt.subplots(rows, 3, figsize=(16, 5 * rows))
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


def eda_visualizacion_suvs(df, target="Precio", current_year=2026):
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