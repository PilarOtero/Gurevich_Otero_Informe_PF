import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

#FUNCIONES DE PREPROCESSING PARA ANTES DEL SPLIT
def convertir_a_usd(dataset, tipo_de_cambio):
    dataset= dataset.copy()
    dataset['Precio'] = np.where(dataset['Moneda'] == '$', dataset['Precio'] / tipo_de_cambio, dataset['Precio'])

    dataset = dataset.drop(columns = ['Moneda'])
    return dataset

def limpiar_col_suv_unnamed(dataset):
    #Borra la columna UNNAMED y TIPO DE CARROCERIA al ser todas las mustras SUV (no aporta info)
    dataset= dataset.copy()
    dataset = dataset.drop(columns=["Unnamed: 0", "Tipo de carrocería"])
    return dataset

def limpiar_filas_motor(dataset):
    #MOTOR -> eliminamos las muestras con valor nulo al ser pocas
    dataset= dataset.copy()
    dataset = dataset.dropna(subset=['Motor'])
    return dataset

def cambiar_5_puertas(dataset):
    #Los valores mayores a 5 para las puertas, los reemplazamos por ese valor asumiendo que tienen el maximo valor posible
    dataset= dataset.copy()
    dataset['Puertas'] = np.where(dataset['Puertas'].isin([2,3,4,5]), dataset['Puertas'], 5)
    return dataset

def pasar_kilometros_numerico(dataset):
    dataset = dataset.copy()
    dataset["Kilómetros"] = pd.to_numeric(dataset["Kilómetros"], errors="coerce")
    return dataset

def crear_0km(dataset):
    dataset = dataset.copy()
    dataset["0km"] = (dataset["Kilómetros"] == 0).astype(int)
    return dataset

def preprocesamiento_pre_split(dataset):
    dataset= dataset.copy()
    dataset = limpiar_col_suv_unnamed(dataset)
    dataset = limpiar_filas_motor(dataset)
    dataset = cambiar_5_puertas(dataset)
    dataset = pasar_kilometros_numerico(dataset)
    dataset = convertir_a_usd(dataset, tipo_de_cambio=1100)
    dataset = crear_0km(dataset)

    return dataset

#FUNCIONES DE PREPROCESSING PARA DESPUES DEL SPLIT -> se usan los valores de train
def moda_color(X_train):
    #Agrupa por marca y modelo y cuenta la cantidad de cada color y ordena de mayor a menor por grupo
    apariciones_color = X_train.groupby(['Marca', 'Modelo'])['Color'].value_counts()
    apariciones_color_dataframe = apariciones_color.reset_index()
    #Conserva solo la primer fila de cada color
    mas_frecuente = apariciones_color_dataframe.drop_duplicates(['Marca', 'Modelo'])

    return mas_frecuente[['Marca', 'Modelo', 'Color']].rename(columns = {'Color': 'Color moda'})

def moda_camara(X_train):
    #Agrupa por marca y año y cuenta la cantidad de cada cámara y ordena de mayor a menor por grupo
    apariciones_camara = X_train.groupby(['Marca', 'Año'])['Con cámara de retroceso'].value_counts()
    apariciones_camara_dataframe = apariciones_camara.reset_index()
    #Conserva solo la primer fila de cada cámara
    mas_frecuente = apariciones_camara_dataframe.drop_duplicates(['Marca', 'Año'])

    return mas_frecuente[['Marca', 'Año', 'Con cámara de retroceso']].rename(columns = {'Con cámara de retroceso': 'Con cámara de retroceso moda'})

def knn_transmision(set, dummy_cols = None):
    null_mask = set['Transmisión'].isna()
    dummies = pd.get_dummies(set['Transmisión'], prefix='Trans_')
    if dummy_cols is not None:
        dummies = dummies.reindex(columns = dummy_cols, fill_value = 0)
    #Las filas que eran nulas las convertimos a NaN en las dummies para que KNN les impute valores
    dummies[null_mask] = np.nan

    return pd.concat([dummies, set[['Año']]], axis = 1), null_mask, dummies.columns.to_list()

def transmision_sets(X_train, X_val):
    #Generar las columnas dummies
    knn_input_train, null_transmision_mask_train, transmision_columns = knn_transmision(X_train)
    knn_input_val, null_transmision_mask_val, _ = knn_transmision(X_val, dummy_cols = transmision_columns)

    #Buscar los 5 vecinos mas parecidos para cada muestra con valores faltantes y les ponemos el valor de la media
    imputer = KNNImputer(n_neighbors = 5)
    #Aprender y aplicar los parametros a train
    imputed_train = imputer.fit_transform(knn_input_train)
    #Aplica los parametros de train a validation
    imputed_val = imputer.transform(knn_input_val)

    #Convertimos a DF
    imputed_train = pd.DataFrame(imputed_train, columns = knn_input_train.columns, index = X_train.index)
    imputed_val = pd.DataFrame(imputed_val, columns = knn_input_val.columns, index = X_val.index)

    #Filtrar las columnas de transmision
    transmision_columns = [col for col in imputed_train.columns if col.startswith('Trans_')]
    #Para cada fila nula, encuentra la columna de transmision con el valor mas alto, reemplaza por ese valor y saca el prefijo Trans_ de la columna para dejarla con el nombre original
    decoded_train_cols = imputed_train.loc[null_transmision_mask_train, transmision_columns].idxmax(axis=1).str.replace('Trans_', '', regex = False)
    decoded_val_cols = imputed_val.loc[null_transmision_mask_val, transmision_columns].idxmax(axis=1).str.replace('Trans_', '', regex = False)
    
    #Volvemos a la columna original, con los valores nulos ya reemplazados
    X_train.loc[null_transmision_mask_train, 'Transmisión'] = decoded_train_cols
    X_val.loc[null_transmision_mask_val, 'Transmisión'] = decoded_val_cols

    return X_train, X_val

def completar_kilometros(X_train, X_val):
    mediana_por_año = X_train.groupby('Año')['Kilómetros'].median()
    mediana_global = X_train['Kilómetros'].median()

    X_train['Kilómetros'] = X_train['Kilómetros'].fillna(X_train['Año'].map(mediana_por_año))

    medianas_val = X_val['Año'].map(mediana_por_año).fillna(mediana_global)
    X_val['Kilómetros'] = X_val['Kilómetros'].fillna(medianas_val)

    return X_train, X_val

#FEATURE ENGINEERING
def crear_features_autos(set, current_year=2026):
    """
    Crea variables nuevas relevantes para explicar el precio.

    - Antiguedad: años de uso aproximados.
    - Km_por_año: intensidad de uso del vehículo.
     """
    set = set.copy()

    set["Antiguedad"] = current_year - set["Año"]
    set["Km_por_año"] = (set["Kilómetros"] / (set["Antiguedad"] + 1))
    
    return set

def preprocesamiento_post_split(X_train, X_val):
    sets = [X_train, X_val]

    #COLOR -> completamos los valores faltantes con la moda por marca + modelo con los datos de TRAIN
    color_mas_frecuente = moda_color(X_train)

    #CAMARA DE RETROCESO -> completamos los valores faltantes con la moda por marca + año (asumiendo que todos los modelos del año de la marca de la muestra tienen o no camara)
    camara_mas_frecuente = moda_camara(X_train)

    for i, dataset in enumerate(sets):
        #COLOR
        #Agrega color_moda como nueva columna a la izquierda
        dataset = dataset.merge(color_mas_frecuente, on = ['Marca', 'Modelo'], how = 'left')
        #Rellena los valores nulos de color con el valor de color_moda
        dataset['Color'] = dataset['Color'].fillna(dataset['Color moda'])
        #Borra la columna agregada
        dataset = dataset.drop(columns=['Color moda'])

        #CAMARA DE RETROCESO
        dataset = dataset.merge(camara_mas_frecuente, on = ['Marca', 'Año'], how = 'left')
        dataset['Con cámara de retroceso'] = dataset['Con cámara de retroceso'].fillna(dataset['Con cámara de retroceso moda'])
        #Borra la columna agregada
        dataset = dataset.drop(columns=['Con cámara de retroceso moda'])

        sets[i] = dataset

    X_train, X_val = sets

    #TRANSMISION -> One-Hot Encoding -> KNN imputer -> decode back
    X_train, X_val = transmision_sets(X_train, X_val)

    #KILOMETROS -> mediana agrupada por año 
    X_train, X_val = completar_kilometros(X_train, X_val)

    #FEATURE ENGINEERING
    X_train = crear_features_autos(X_train)
    X_val = crear_features_autos(X_val)

    return X_train, X_val

#################################################
#ONE HOT LUEGO DE TODO EL PREPROCESSING 
def onehot_encoding(X_train, X_val, categoricas) -> pd.DataFrame:
    """
    Aplica OneHot Encoding a las variables categoricas

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features para entrenamiento
            X_val(pd.DataFrame): matriz de features para validación

        Parámetros de salida:
            X_train(pd.DataFrame): matriz de features para entrenamiento luego de la aplicación de OneHot Encoding 
            X_val(pd.DataFrame): matriz de features para validación luego de la aplicación de OneHot Encoding 
    """
    encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')

    #Aprender las categorias de Train
    encoder.fit(X_train[categoricas])
    #Generar los nombres de las columnas dummy
    nuevas_columnas = encoder.get_feature_names_out(categoricas)

    #Modificar train y validation con las nuevas columnas
    train_encoded = pd.DataFrame(encoder.transform(X_train[categoricas]), columns = nuevas_columnas, index = X_train.index)
    val_encoded = pd.DataFrame(encoder.transform(X_val[categoricas]), columns = nuevas_columnas, index = X_val.index)

    #Reemplazar las columnas originales por las nuevas
    X_train = pd.concat([X_train.drop(columns = nuevas_columnas), train_encoded], axis = 1)
    X_val = pd.concat([X_val.drop(columns = nuevas_columnas), val_encoded], axis = 1)

    return X_train, X_val
