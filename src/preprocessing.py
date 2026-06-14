import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def moda_color(dataset):
    #Agrupa por marca y modelo y cuenta la cantidad de cada color y ordena de mayor a menor por grupo
    apariciones_color = dataset.groupby(['Marca', 'Modelo'])['Color'].value_counts()
    apariciones_color_dataframe = apariciones_color.reset_index()
    #Conserva solo la primer fila de cada color
    mas_frecuente = apariciones_color_dataframe.drop_duplicates(['Marca', 'Modelo'])

    return mas_frecuente[['Marca', 'Modelo', 'Color']].rename(columns = {'Color': 'Color moda'})

def moda_camara(dataset):
    #Agrupa por marca y año y cuenta la cantidad de cada cámara y ordena de mayor a menor por grupo
    apariciones_camara = dataset.groupby(['Marca', 'Año'])['Con cámara de retroceso'].value_counts()
    apariciones_camara_dataframe = apariciones_camara.reset_index()
    #Conserva solo la primer fila de cada cámara
    mas_frecuente = apariciones_camara_dataframe.drop_duplicates(['Marca', 'Año'])

    return mas_frecuente[['Marca', 'Año', 'Con cámara de retroceso']].rename(columns = {'Con cámara de retroceso': 'Con cámara de retroceso moda'})

def convertir_a_usd(dataset, tipo_de_cambio):
    dataset['Precio'] = np.where(dataset['Moneda'] == '$', dataset['Precio'] / tipo_de_cambio, dataset['Precio'])

    dataset = dataset.drop(columns = ['Moneda'])
    return dataset

def handle_values(dataset, tipo_de_cambio:int = 1100):
    #MOTOR -> eliminamos las muestras con valor nulo al ser pocas
    dataset = dataset.dropna(subset=['Motor'])

    #COLOR -> completamos los valores faltantes con la moda por marca + modelo
    color_mas_frecuente = moda_color(dataset)
    #Agrega color_moda como nueva columna a la izquierda
    dataset = dataset.merge(color_mas_frecuente, on = ['Marca', 'Modelo'], how = 'left')
    #Rellena los valores nulos de color con el valor de color_moda
    dataset['Color'] = dataset['Color'].fillna(dataset['Color moda'])

    #Borra la columna agregada
    dataset = dataset.drop(columns=['Color moda'])

    #CAMARA DE RETROCESO -> completamos los valores faltantes con la moda por marca + año (asumiendo que todos los modelos del año de la marca de la muestra tienen o no camara)
    camara_mas_frecuente = moda_camara(dataset)
    dataset = dataset.merge(camara_mas_frecuente, on = ['Marca', 'Año'], how = 'left')
    dataset['Con cámara de retroceso'] = dataset['Con cámara de retroceso'].fillna(dataset['Con cámara de retroceso moda'])

    #Borra la columna agregada
    dataset = dataset.drop(columns=['Con cámara de retroceso moda'])

    #TRANSMISION -> One-Hot Encoding -> KNN imputer -> decode back
    null_transmision_mask = dataset['Transmisión'].isna()
    dummies = pd.get_dummies(dataset['Transmisión'], prefix='Trans_')
    #Las filas que eran nulas las convertimos a NaN en las dummies para que KNN les impute valores
    dummies[null_transmision_mask] = np.nan

    #Agregamos la columna año asumiendo que los modelos del mismo año y marca tienen la misma transmisión
    knn_input = pd.concat([dummies, dataset[['Año']]], axis=1)
    #Buscamos los 5 vecinos mas parecidos para cada muestra con valores faltantes y les ponemos el valor de la media
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(knn_input)
    #Convertimos a DF
    imputed_dataset = pd.DataFrame(imputed_data, columns = knn_input.columns, index = dataset.index)

    #Filtra las columnas de transmision
    transmision_columns = [col for col in imputed_dataset.columns if col.startswith('Trans_')]
    #Para cada fila nula, encuentra la columna de transmision con el valor mas alto, reemplaza por ese valor y saca el prefijo Trans_ de la columna para dejarla con el nombre original
    decoded_columns = imputed_dataset.loc[null_transmision_mask, transmision_columns].idxmax(axis=1).str.replace('Trans_', '', regex = False)
    #Volvemos a la columna original, con los valores nulos ya reemplazados
    dataset.loc[null_transmision_mask, 'Transmisión'] = decoded_columns

    #Borra la columna UNNAMED y TIPO DE CARROCERIA al ser todas las mustras SUV (no aporta info)
    dataset = dataset.drop(columns=['Unnamed: 0'])
    dataset = dataset.drop(columns=['Tipo de carrocería'])

    #Los valores mayores a 5 para las puertas, los reemplazamos por ese valor asumiendo que tienen el maximo valor posible
    dataset['Puertas'] = np.where(dataset['Puertas'].isin([2,3,4,5]), dataset['Puertas'], 5)

    #KILOMETROS
    dataset['0km'] = (dataset['Kilómetros'] == '0').astype(int)

    #MONEDA
    dataset = convertir_a_usd(dataset, tipo_de_cambio)

    return dataset


    

    