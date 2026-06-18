import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

#FUNCIONES DE PREPROCESSING PRE SPLIT
def convertir_a_usd(dataset:pd.DataFrame, tipo_de_cambio:float):
    """
    Convierte el precio de las muestras de SUV que se definen en pesos a dolares 

        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja
            tipo_de_cambio(float): tipo de cambio ARS/USD utilizado para la conversión peso a dólar

        Parámetros de salida:
            dataset(pd.DataFrame): dataset con la modificación realizada
        
    """
    dataset= dataset.copy()
    dataset['Precio'] = np.where(dataset['Moneda'] == '$', dataset['Precio'] / tipo_de_cambio, dataset['Precio'])
    
    dataset = dataset.drop(columns = ['Moneda'])
    
    return dataset

def limpiar_cols(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Elimina las columnas 'Unnamed:0', 'SUV', 'Título' y 'Tipo de Carroceria' del dataset

        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja
    
        Parámetros de salida:
            dataset(pd.DataFrame): dataset con la modificación realizada
    """
    dataset= dataset.copy()
    #Borrar la columna UNNAMED y TIPO DE CARROCERIA al ser todas las mustras SUV (no aporta info)
    dataset = dataset.drop(columns=["Unnamed: 0", "Tipo de carrocería", "Título"])
    return dataset

def limpiar_filas_motor(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Elimina las filas con valores nulos en la columna 'Motor' al ser pocas muestras en las que esto ocurre
    
        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja
    
        Parámetros de salida:
            dataset(pd.DataFrame): dataset con la modificación realizada
    """
    dataset= dataset.copy()
    dataset = dataset.dropna(subset=['Motor'])
    return dataset

def tratar_motor(dataset:pd.DataFrame)-> pd.DataFrame:
    dataset= dataset.copy()
    extraido = (
        dataset['Motor']
        .str.extract(r'(\d+[.,]\d+)')[0]
        .str.replace(',', '.')
        .astype(float))    
    dataset['Motor'] = extraido.combine_first(dataset['Motor'])
    #dataset = dataset.dropna(subset=['Motor']) 
    return dataset

def corregir_marcas(dataset:pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset['Marca'] = dataset['Marca'].replace({'Hiunday': 'Hyundai', 
      'hiunday': 'Hyundai', 
      'Rrenault': 'Renault', 
      'Jetur': 'Jetour',
      'Vol': 'Volkswagen', 
      'D-S': 'D.S', 
      'DS AUTOMOBILES': 'D.S', 
      'Range Rover': 'Land Rover'})
    
    return dataset

def analizar_puertas(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza los valores inválidos de la columna 'Puertas' por 5 en el caso donde puertas = 4, elimina las muestras con 2 puertas
        
        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja
    
        Parámetros de salida:
            dataset(pd.DataFrame): dataset con la modificación realizada
    """
    dataset= dataset.copy()
    dataset = dataset[dataset['Puertas'] != 2]
    dataset['Puertas'] = np.where(dataset['Puertas'].isin([3,5]), dataset['Puertas'], 5)
    return dataset

def pasar_kilometros_numerico(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la columna 'Kilómetros' a numérica, reemplazando por NaN los valores que no puedan convertirse
        
        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja
    
        Parámetros de salida:
            dataset(pd.DataFrame): dataset con la modificación realizada
    """
    dataset = dataset.copy()
    dataset["Kilómetros"] = pd.to_numeric(dataset["Kilómetros"], errors="coerce")
    return dataset

def crear_0km(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna '0km' que indica si el vehículo es nuevo (1 si los kilómetros son 0, y 0 en caso contrario)
        
        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja
    
        Parámetros de salida:
            dataset(pd.DataFrame): dataset con la modificación realizada
    """
    dataset = dataset.copy()
    dataset["0km"] = (dataset["Kilómetros"] == 0).astype(int)
    return dataset

def descripcion_scoring(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Genera un score numerico del 1 al 10 para cada muestra en base a su descripcion.
    Asigna +1 por cada palabra positiva hallada y -1 por cada palabra negativa.
    Se normaliza el score al rango [1,10] y se clippea en esos valores maximos.
    Se elimina la columna 'Descripción'
    
        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja
    
        Parámetros de salida:
            dataset(pd.DataFrame): dataset con la modificación realizada
    """
    dataset = dataset.copy()
    descripcion = dataset['Descripción'].str.lower()

    palabras_positivas = [
        # Originales
        'único dueño', 'unico dueño', 'única mano', 'unica mano', 'primera mano', '1ra mano',
        'service oficial', 'service al día', 'services completos',
        'impecable', 'excelente estado', 'impoluto',
        'cubiertas nuevas', 'garantía', 'listo para transferir', '0km', 'gomas nuevas',
        'papeles al día', 'documentación al día', 'vtv al día', 'vtv vigente', 'vtv apto',
        'km reales', 'guardado en cochera', 'nunca un golpe', 'sin detalles',
        'distribución recién', 'cadena nueva', 'batería nueva',
        # Forma femenina de dueño único (ej: "Unica dueña")
        'única dueña', 'unica dueña',
        # Sin tildes (vendedores suelen omitirlas)
        'garantia', 'papeles al dia', 'documentacion al dia', 'vtv al dia', 'bateria nueva',
        # Femenino / variantes de "listo para transferir"
        'lista para transferir', 'lista para la transferencia', 'listo para la transferencia',
        # Variantes de km reales
        'kilometraje real', 'kilómetros reales', 'kilometros reales',
        # Variantes de service / servicios
        'servicios al día', 'servicios al dia', 'servicio oficial', 'servicios oficiales',
        'todos los service', 'servis oficiales', 'servís oficiales',
        # Estado general del vehículo
        'buen estado', 'muy buen estado', 'perfecto estado',
        'muy cuidado', 'muy cuidada', 'bien cuidado', 'bien cuidada',
        'no hay que hacerle nada',
        # Funcionamiento
        'funciona todo', 'todo funciona', 'funcionando perfectamente',
        # Cobertura mecánica / garantía de concesionario
        'cobertura mecánica', 'cobertura mecanica',
        # Neumáticos nuevos (variantes)
        'neumáticos nuevos', 'neumaticos nuevos',
        # Señales de calidad de agencia (femenino incluido)
        'concesionario oficial', 'concesionaria oficial', 'agencia oficial', 'usados seleccionados',
        # Estado excelente / variantes
        'inmaculada', 'inmaculado', 'inmejorable estado',
        'muy buenas condiciones', 'buenas condiciones',
        # Documentación
        'papeles en regla', 'titular al día', 'titular al dia',
        # Sin tilde adicional de service
        'service al dia',
        # Cochera / garage
        'guardado en garage', 'guardado en garaje', 'siempre en cochera',
        # VTV femenino
        'vtv apta',
        # Servicios completos variantes
        'todos los servicios',
        # Dueño / titular
        'solo dueño', 'único titular', 'unico titular', 'segundo dueño',
        # Cubiertas / gomas casi nuevas
        'casi nuevas', 'casi nuevo',
        # Funcionamiento (variante sin -ndo)
        'funciona perfectamente',
        # Entrega inmediata (stock físico disponible)
        'entrega inmediata',
        'poco uso',
        'cuotas',
        'muy bueno', 'nada para hacerle'
    ]
    palabras_negativas = [
        # Originales
        'no incluyó una descripción', 'chocado', 'reparado', 'motor no funciona',
        'detalle de chapa', 'detalle a la vista',
        # Plural de "detalle a la vista" (ej: "detalles a la vista casoletas...")
        'detalles a la vista',
        # Daños de carrocería
        'abolladura', 'abolladuras', 'leve choque',
        'detalle de pintura', 'detalles de pintura',
        'parabrisas rajado', 'parabrisas roto',
        'rayado', 'rayones',
        # Necesita reparación
        'hay que reparar', 'hay q reparar', 'necesita reparacion', 'necesita reparación',
        'falta arreglar', 'para reparar', 'a reparar',
        # GNC (reduce valor de reventa)
        'con gnc', 'tiene gnc', 'gnc instalado',
        # Otros negativos
        'accidentado', 'siniestrado', 'motor golpeado',
        # Detalles cosméticos / estéticos
        'detalles estéticos', 'detalles esteticos', 'algunos detalles'
    ]

    #Por cada palabra positiva de la descripcion se convierte en 0/1 y se suman todas en score descripcion
    descripciones_positivas = sum(descripcion.str.contains(positivas).astype(int) for positivas in palabras_positivas)
    descripciones_negativas = sum(descripcion.str.contains(negativas).astype(int) for negativas in palabras_negativas)

    score_raw = descripciones_positivas - descripciones_negativas

    #Definicion rango [1,10]
    dataset['Score Descripción'] = score_raw.clip(lower = 0)
    #Se divide por el maximo del dataset (queda entre 0-1) y se multiplica por 9 para que quede entre 0-9
    dataset['Score Descripción'] = 1 + (dataset['Score Descripción'] / dataset['Score Descripción'].max()) * 9
    dataset['Score Descripción'] = dataset['Score Descripción'].clip(upper = 10).round().astype(int)

    dataset = dataset.drop(columns = ['Descripción'])

    return dataset 

def flags_version(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Extrae keywords de la columna 'Versión' y crea columnas binarias (0/1) por cada una.
    Elimina la columna original 'Versión'.

        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja

        Parámetros de salida:
            dataset(pd.DataFrame): dataset con columnas version_<keyword> y sin 'Versión'
    """
    dataset = dataset.copy()

    keywords = [
        "limited", "sport", "sr", "srv", "titanium", "executive",
        "highline", "comfortline", "comfort", "trendline", "premium", "elite",
        "rubicon", "laredo", "overland",
        "4x4", "4wd", "awd", "at", "at6", "mt", "cvt", "turbo", "tdi", "tsi",
        "plus", "pro", "pack", "fwd",
        "ltz", "exclusive", "feline", "allure", "advance", "longitude",
        "sense", "xei", "active", "gt",
        "se", "freestyle", "privilege", "ph2", "dynamique", "zen",
        "intens", "feel", "4matic", "expression", "xls",
        "confort", "xdrive", "xline", "tfsi", "sxt", "sel", "amg",
        "luxury", "techo", "luxe", "life", "intelligent",
        "ex", "vx", "crossway", "classic", "s", "gl",
        "v6", "lt", "prado",
        "pop", "lx", "live", "rock"
    ]

    keywords_exactos = {
        "at", "at6", "mt", "sr", "srv", "awd", "4x4", "4wd", "pro", "gt",
        "fwd", "ltz", "xei", "se", "ph2", "zen", "xls", "xline", "sxt", "sel", "amg",
        "life", "ex", "vx", "s", "gl", "v6", "lt", "pop", "lx"
    }

    for kw in keywords:
        patron = rf"\b{kw}\b" if kw in keywords_exactos else kw
        dataset[f"version_{kw}"] = (
            dataset["Versión"].str.lower().str.contains(patron, na=False).astype(int)
        )

    dataset = dataset.drop(columns=["Versión"])
    return dataset

def unir_colores(dataset):
    dataset = dataset.copy()
    dataset['Color'] = dataset['Color'].str.lower().replace({
        # Blanco
        'blanca': 'blanco', 'summit white': 'blanco', 'mineralweiss metallic': 'blanco', 'blanco nacre tricapa': 'blanco', 'blanco banquise': 'blanco', 
        'blanco banchisa bicolor negro': 'blanco', 'blanco glaciar': 'blanco',
        # Negro
        'negra': 'negro', 'carbon black': 'negro', 'black meet kettle': 'negro', 'noir perla nera': 'negro',
        # Gris
        'gris oscuro': 'gris', 'acero': 'gris', 'grafito': 'gris', 'gray': 'gris', 'cendre': 'gris', 'gris plata': 'gris', 'gris selenium': 'gris', 'gris artense': 'gris', 
        'gris titane': 'gris', 'gris laque': 'gris', 'gris indy': 'gris', 'gris silverstone': 'gris', 'gris estrella': 'gris', 'gris platino': 'gris', 'granite crysta bc': 'gris', 
        'granite crystal bc': 'gris', 'skyscraper grau metallic': 'gris',
        # Plateado
        'plata': 'plateado', 'plata bari': 'plateado', 'prata bari+tet vulc': 'plateado',
        # Rojo
        'rojo sunset metalizado': 'rojo',
        # Azul
        'blue': 'azul', 'steel_blue': 'azul', 'celeste': 'azul',
        # Marrón
        'marrón oscuro': 'marrón', 'marrón claro': 'marrón', 'marron kodiak': 'marrón',
        # Beige
        'beige techo negro': 'beige', 'café': 'beige',
        # Dorado
        'champaing': 'dorado', 'cobre': 'dorado',
        # Verde
        'verde oscuro': 'verde',
        # Violeta
        'morado': 'violeta', 'morado oscuro': 'violeta',
        # Amarillo
        'amarrillo': 'amarillo',
        # Otro
        'moundaz': 'otro',
    })
    return dataset

def preprocesamiento_pre_split(dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el preprocesamiento inicial al dataset completo, antes de realizar el split en entrenamiento y validación
    (por eso se aplican transformaciones basadas únicamente en los datos de cada muestra, sin riesgo de data leakeage)
    
        Parámetros de entrada:
            dataset(pd.DataFrame): dataset sobre el que se trabaja
    
        Parámetros de salida:
            dataset(pd.DataFrame): dataset con la modificación realizada
    """
    dataset = dataset.copy()

    #Considerando que las concesionarias ya tienen modelos de 2025 a la venta
    dataset = dataset[dataset['Año'] <= 2025]
    dataset = limpiar_cols(dataset)
    dataset = limpiar_filas_motor(dataset)
    dataset = tratar_motor(dataset)
    dataset = corregir_marcas(dataset)
    dataset = analizar_puertas(dataset)
    dataset = pasar_kilometros_numerico(dataset)
    #Definimos el tipo de cambio promedio de mayo 2024 (fecha del dataset)
    dataset = convertir_a_usd(dataset, tipo_de_cambio = 884.60)
    dataset = crear_0km(dataset)
    dataset = descripcion_scoring(dataset)
    dataset = flags_version(dataset)
    dataset = unir_colores(dataset)

    return dataset

#FUNCIONES DE PREPROCESSING POST SPLIT -> se usan los valores de train
def moda_color(X_train:pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el color más frecuente por combinación de 'Marca' y 'Modelo' (sobre el set de entrenamiento) para imputar valores nulos de la columna 'Color'

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features de entrenamiento

        Parámetros de salida:
            (pd.DataFrame): DataFrame con las columnas 'Marca', 'Modelo', 'Color moda' 
    """
    #Agrupa por marca y modelo y cuenta la cantidad de cada color y ordena de mayor a menor por grupo
    apariciones_color = X_train.groupby(['Marca', 'Modelo'])['Color'].value_counts()
    apariciones_color_dataframe = apariciones_color.reset_index()
    #Conserva solo la primer fila de cada color
    mas_frecuente = apariciones_color_dataframe.drop_duplicates(['Marca', 'Modelo'])

    return mas_frecuente[['Marca', 'Modelo', 'Color']].rename(columns = {'Color': 'Color moda'})

def moda_camara(X_train:pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el valor más frecuente de 'Con cámara de retroceso' por combinación de 'Marca' y 'Año' (sobre el set de entrenamiento) para imputar valores nulos

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features de entrenamiento

        Parámetros de salida:
            (pd.DataFrame): DataFrame con las columnas 'Marca', 'Año', 'Con cámara de retroceso moda' 
    """
    #Agrupa por marca y año y cuenta la cantidad de cada cámara y ordena de mayor a menor por grupo
    apariciones_camara = X_train.groupby(['Marca', 'Año'])['Con cámara de retroceso'].value_counts()
    apariciones_camara_dataframe = apariciones_camara.reset_index()
    #Conserva solo la primer fila de cada cámara
    mas_frecuente = apariciones_camara_dataframe.drop_duplicates(['Marca', 'Año'])

    return mas_frecuente[['Marca', 'Año', 'Con cámara de retroceso']].rename(columns = {'Con cámara de retroceso': 'Con cámara de retroceso moda'})

def knn_transmision(set:pd.DataFrame, dummy_cols:list = None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepara la columna de 'Transmisión' para ser imputada con KNNImputer
    Aplica get_dummies y marca como NaN las filas que originalmente eran nulas

        Parámetros de entrada:
            set(pd.DataFrame): subset del dataset 
            dummy_cols(list): columnas dummy esperadas para alinear validación con entrenamiento (None si es entrenamiento)
    
        Parámetros de salida:
            (pd.DataFrame): columnas dummy de transmisión + columna 'Año' para el KNN
            null_mask(pd.Series): mascara booleana con True donde 'Transmisión' era nula
            (list de str): nombres de las columnas dummy generadas
    """
    null_mask = set['Transmisión'].isna()
    dummies = pd.get_dummies(set['Transmisión'], prefix='Trans_')
    if dummy_cols is not None:
        dummies = dummies.reindex(columns = dummy_cols, fill_value = 0)
    #Las filas que eran nulas las convertimos a NaN en las dummies para que KNN les impute valores
    dummies = dummies.astype(float)
    dummies[null_mask] = np.nan

    return pd.concat([dummies, set[['Año']]], axis = 1), null_mask, dummies.columns.to_list()

def transmision_sets(X_train:pd.DataFrame, X_val:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputa los valores nulos de 'Transmisión' usando KNNImputer sobre las columnas dummy
    Aprende los parámetros sobre train y los aplica a ambos sets
    Decodifica el resultado del KNN al valor categórico original
    
        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features para entrenamiento
            X_val(pd.DataFrame): matriz de features para validación

        Parámetros de salida:
            X_train(pd.DataFrame): matriz de entrenamiento con 'Transmisión' sin nulos
            X_val(pd.DataFrame): matriz de validación con 'Transmisión' sin nulos
    """
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

def completar_kilometros(X_train:pd.DataFrame, X_val:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputa los valores nulos de 'Kilómetros' usando la mediana agrupada por 'Año'
    Aprende las medianas sobre train y las aplica a ambos sets.
    Si un año de validación no existe en train, se usa la mediana global de train
        
        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features para entrenamiento
            X_val(pd.DataFrame): matriz de features para validación

        Parámetros de salida:
            X_train(pd.DataFrame): matriz de entrenamiento con 'Kilómetros' sin nulos
            X_val(pd.DataFrame): matriz de validación con 'Kilómetros' sin nulos
    """
    mediana_por_año = X_train.groupby('Año')['Kilómetros'].median()
    mediana_global = X_train['Kilómetros'].median()

    X_train['Kilómetros'] = X_train['Kilómetros'].fillna(X_train['Año'].map(mediana_por_año))

    medianas_val = X_val['Año'].map(mediana_por_año).fillna(mediana_global)
    X_val['Kilómetros'] = X_val['Kilómetros'].fillna(medianas_val)

    return X_train, X_val

#FEATURE ENGINEERING
def crear_features_autos(set:pd.DataFrame, año_actual:int = 2024) -> pd.DataFrame:
    """
    Crea variables nuevas relevantes para explicar el precio.
    - Antiguedad: años de uso aproximados
    - Km_por_año: intensidad de uso del vehículo

        Parámetros de entrada:
            set(pd.DataFrame): subset del dataset sobre el que se generarán los nuevos features
            año_actual(int): año actual 

        Parámetros de salida:
            set(pd.DataFrame): subset con las nuevas variables
     """
    set = set.copy()

    set["Antiguedad"] = (año_actual - set["Año"]).clip(lower = 0)
    set["Km_por_año"] = (set["Kilómetros"] / (set["Antiguedad"] + 1))
    
    return set

def preprocesamiento_post_split(X_train:pd.DataFrame, X_val:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica el preprocesamiento de los datos post split en entrenamiento y validación para la utilización de los parámetros del primero sobre ambos sets.
    Se completan los valores faltantes para el color, cámara de retroceso, transmisión y kilómetros. Se aplica Feature Engineering para futuros usos.

        Parámetros de entrada:
            X_train(pd.DataFrame): matriz de features para entrenamiento
            X_val(pd.DataFrame): matriz de features para validación
       
        Parámetros de salida:
            X_train(pd.DataFrame): matriz de features para entrenamiento luego del preprocesamiento
            X_val(pd.DataFrame): matriz de features para validación luego del preprocesamiento
    """
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

#ONE-HOT LUEGO DE TODO EL PREPROCESSING 
def onehot_encoding(X_train, X_val, categoricas) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    X_train = pd.concat([X_train.drop(columns = categoricas), train_encoded], axis = 1)
    X_val = pd.concat([X_val.drop(columns = categoricas), val_encoded], axis = 1)

    return X_train, X_val
