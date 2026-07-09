# Gurevich_Otero_Informe_PF

## ESTRUCTURA DEL PROYECTO

Gurevich_Otero_Informe_PF/
├── data/
│   ├── raw/
│   │   ├── pf_suvs.csv
│   │   └── SUVS_2026-test-masked.csv
│   └── processed/
│       ├── data_pre.csv
│       ├── X_train_EDA.csv
│       └── X_val_EDA.csv
├── notebook/
│   ├── Gurevich_Otero_Notebook_PF.ipynb
│   └── artefactos/
│       ├── modelo_rl_base.pkl
│       ├── modelo_rl_final.pkl
│       ├── modelo_nn.pkl
│       ├── modelo_xgboost.pkl
│       ├── tfidf.pkl
│       ├── svd.pkl
│       ├── ohe.pkl
│       └── scaler.pkl
├── artefactos_finales/
│   ├── modelo_final_xgboost.pkl
│   ├── tfidf_final.pkl
│   └── svd_final.pkl
├── src/
│   ├── data_splitting.py
│   ├── deteccion_outliers.py
│   ├── metrics.py
│   ├── plots.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── modelos/
│       ├── regresion_lineal.py
│       ├── xgboost.py
│       └── red_neuronal.py
├── Gurevich_Otero_XGBoost_predictions.csv
└── Gurevich_Otero_Predictions_PF_SUV.zip

### Datos
- **data/raw/pf_suvs.csv** → Dataset original de desarrollo, con publicaciones de SUVs usadas en Argentina. Contiene variables descriptivas del vehículo (marca, modelo, año, kilometraje, motor, transmisión, color, tipo de vendedor, descripción del aviso, entre otras) y la variable objetivo *Precio*.
- **data/raw/SUVS_2026-test-masked.csv** → Conjunto de test provisto por la cátedra, utilizado para generar las predicciones finales con el modelo seleccionado.
- **data/processed/data_pre.csv** → Dataset resultante de la limpieza y el preprocesamiento previo al split (unificación de marcas, tratamiento de motor, colores, cámara de retroceso, etc.).
- **data/processed/X_train_EDA.csv / X_val_EDA.csv** → Conjuntos de entrenamiento y validación ya con One-Hot Encoding y TF-IDF + SVD aplicados sobre la descripción, utilizados para el entrenamiento de los modelos.

### Notebook
- **notebook/Gurevich_Otero_Notebook_PF.ipynb** → Notebook principal de entrega. Contiene el análisis exploratorio de datos (EDA), la detección y el tratamiento de outliers, el pipeline de preprocesamiento, el entrenamiento y la comparación de los modelos (Regresión Lineal, XGBoost y Red Neuronal), la selección del modelo final y la generación de las predicciones sobre el conjunto de test.
- **notebook/artefactos/** → Modelos y transformaciones (encoder, scaler, TF-IDF, SVD) guardados durante la ejecución de la notebook, utilizados para replicar el preprocesamiento sobre el conjunto de test.

### Artefactos finales
- **artefactos_finales/** → Modelo final seleccionado (XGBoost Categórico con TF-IDF) y las transformaciones asociadas (TF-IDF y SVD), listos para generar predicciones sobre nuevos datos sin necesidad de volver a entrenar.

### Código fuente

- **src/data_splitting.py** → Función para dividir el dataset en conjuntos de entrenamiento y validación (`train_val_split`).

- **src/deteccion_outliers.py** → Funciones para la detección de outliers de precio por grupo (Marca_Modelo) utilizando el método del rango intercuartil (IQR), reporte de los grupos con mayor cantidad de outliers, visualización de los registros flageados, y eliminación de outliers tanto por grupo como por un corte de precio fijo.

- **src/metrics.py** → Implementación de las métricas de regresión utilizadas: MSE, RMSE, MAE y R², junto con funciones que calculan y resumen estas métricas sobre train y validación para los modelos clásicos y la red neuronal.

- **src/plots.py** → Funciones de visualización: distribución de variables por categoría, EDA general de las SUVs, relación entre precio y antigüedad/kilometraje, dispersión de precios por marca, boxplots por grupo, comparación de RMSE train vs validación y curvas de aprendizaje para XGBoost y la Red Neuronal.

- **src/preprocessing.py** → Pipeline completo de limpieza y preprocesamiento: conversión de moneda a USD, limpieza de columnas, unificación de marcas y de colores, tratamiento de motor y de cámara de retroceso, clasificación de versión del vehículo, imputación de valores faltantes (color, transmisión, kilometraje, motor) calculada sobre el set de entrenamiento, incorporación de TF-IDF + SVD sobre la descripción del aviso y One-Hot Encoding de las variables categóricas.

- **src/utils.py** → Funciones de propósito general: estandarización de features (`estandarizar`), preprocesamiento del conjunto de test enmascarado, preparación de datos para XGBoost categórico, entrenamiento del modelo final sobre el dataset completo y generación del archivo de entrega con las predicciones sobre el conjunto de test.

- **src/modelos/regresion_lineal.py** → Entrenamiento de modelos de Regresión Lineal (baseline, Ridge y Lasso) y búsqueda del mejor parámetro de regularización.

- **src/modelos/xgboost.py** → Entrenamiento de un modelo XGBoost, soportando tanto variables categóricas nativas como datos codificados con One-Hot Encoding (parámetro `categorico`), y búsqueda de hiperparámetros (cantidad de árboles, profundidad y tasa de aprendizaje) mediante validación cruzada K-Fold.

- **src/modelos/red_neuronal.py** → Implementación de una red neuronal (MLP) en PyTorch con activaciones configurables, batch normalization, dropout y early stopping, junto con una función de búsqueda que entrena y compara múltiples configuraciones de arquitectura.

## Gráficos y visualización
Todas las visualizaciones se construyeron utilizando `matplotlib.pyplot` y `seaborn`, siguiendo el estilo visual establecido en los tutoriales de la materia.

Claude (el asistente de IA de Anthropic) se utilizó como herramienta de apoyo a lo largo del proyecto — tanto para la configuración del entorno de trabajo (creación del kernel de Jupyter y resolución de errores de entorno) como para mejorar la claridad, estructura y redacción de los análisis escritos derivados de los resultados experimentales.
