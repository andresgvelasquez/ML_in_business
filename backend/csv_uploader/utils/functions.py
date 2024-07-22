# Importaciones para la gestion de credenciales
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib as plt
import numpy as np

# Configura el logging
logging.basicConfig(level=logging.INFO)

def save_to_postgres(df, table_name):

    # Utiliza las configuraciones de settings.py
    user = 'admin'
    password = '12345'
    host = 'db'
    database = 'django_web'

    # Crea la cadena de conexión
    connection_string = f'postgresql+psycopg2://{user}:{password}@{host}/{database}'

    # Crea el motor de SQLAlchemy
    engine = create_engine(connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Guarda el DataFrame en PostgreSQL
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        logging.info(f'DataFrame guardado en la tabla {table_name} en PostgreSQL.')
        session.commit()  # Commit explícito
    except Exception as e:
        session.rollback()
        logging.info(f'Error al guardar DataFrame en PostgreSQL: {str(e)}')
    finally:
        session.close()

def generate_summary(df):
    # Tabla 1: Información general
    num_rows = df.shape[0]
    num_duplicates = df.duplicated().sum()
    
    general_summary_json = {
        'Number of Rows': num_rows,
        'Number of Duplicates': num_duplicates
    }

    # # Tabla 2: Información por columna
    df.drop('id', axis=1, inplace=True)
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / num_rows) * 100
    data_types = df.dtypes
    
    # # Generar descripción estadística
    describe_df = df.describe().transpose()  # incluye todas las columnas
    
    # # Agregar la información de describe a la tabla de resumen por columna
    # Tabla 2: Información por columna
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / num_rows) * 100
    data_types = df.dtypes
    
    # Descripción estadística
    describe_df = df.describe(include='all').transpose()
    
    # Convertir la descripción estadística a un formato serializable
    column_summary_json = []
    for column in df.columns:
        col_desc = describe_df.loc[column]
        column_summary_json.append({
            'Column': column,
            'Missing Values': missing_values[column],
            'Missing Percentage (%)': missing_percentage[column],
            'Data Type': str(data_types[column]),
            'Count': col_desc.get('count', 'N/A'),
            'Mean': col_desc.get('mean', 'N/A'),
            'Std Dev': col_desc.get('std', 'N/A'),
            'Min': col_desc.get('min', 'N/A'),
            '25%': col_desc.get('25%', 'N/A'),
            '50%': col_desc.get('50%', 'N/A'),
            '75%': col_desc.get('75%', 'N/A'),
            'Max': col_desc.get('max', 'N/A')
        })

    return general_summary_json , column_summary_json


def data_viz_overview(df):
    df_viz = df.copy()
    
    # Asignar los datos a los bins
    df_viz['product_bins'] = pd.cut(df_viz['product'], bins=10, retbins=False)
    
    # Contar la cantidad de valores en cada bin
    bin_counts = df_viz['product_bins'].value_counts(sort=False, normalize=False)
    
    # Convertir los resultados en un DataFrame
    bin_counts_df = bin_counts.reset_index()
    bin_counts_df.columns = ['Bin', 'Count']
    
    # Convertir a formato JSON
    bin_counts_json = bin_counts_df.to_json(orient='records')  # 'records' para lista de diccionarios
    
    # Calcular los datos para realizar un boxplot
    boxplot_data = {}

    Q1 = df['product'].quantile(0.25)
    Q3 = df['product'].quantile(0.75)
    IQR = Q3 - Q1
    min_val = df['product'].min()
    max_val = df['product'].max()
    median = df['product'].median()
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['product'] < lower_bound) | (df['product'] > upper_bound)]['product'].tolist()

    boxplot_data['product'] = {
        'min': min_val,
        'Q1': Q1,
        'median': median,
        'Q3': Q3,
        'max': max_val,
        'outliers': outliers
    }

    return bin_counts_json, boxplot_data

def featureScaler(features):
    ''' 
    Convierte un dataframe con características númericas
    en un dataframe estandarizado.
    in: dataframe númerico
    out: dataframe estándar
    '''
    scaler = StandardScaler()                                                   
    features_scaled = scaler.fit_transform(features)                            
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)   
    return features_scaled

def extractFeatTarget(data, target_col):
    ''' 
    Divide el datframe en un dataframe con características y otro 
    con el objetivo.
    '''
    features = data.drop(target_col, axis=1)                            
    target = data[target_col]                               
    return features, target

def train_valid_split_scaled(data, target_col, test_size, head=True):
    '''
    Toma un dataframe, las columnas de características y la columna objetivo. Lo divide
    en 2 conjuntos de entrenamiento y 2 conjuntos de validación (características y objetivo). 
    La división depende del test_size (valor entre 0 - 1) y se aplica al conjunto de validación.
    '''
    features, target = extractFeatTarget(data, target_col)
    features_train, features_valid, target_train, target_valid = train_test_split(features, 
                                                                                  target, 
                                                                                  test_size=test_size, 
                                                                                  random_state=54321)
    features_train_scaled = featureScaler(features_train)
    features_valid_scaled = featureScaler(features_valid)
    if head:
       logging.info(features_valid_scaled.head(3))
    return features_train_scaled, target_train, features_valid_scaled, target_valid

def LinearReg_predict(features_train, target_train, features_valid, target_valid):
    ''' 
    Usando un modelo de regresión lineal. Toma las características y objetivos de los
    distintos conjuntos de entrenamiento y validación y devuelve las predicciones para
    el conjuto de validación. 
    '''
    # Inicializar la instancia de regresión lineal
    model = LinearRegression(n_jobs=-1)

    # Definir hiperparámetros a mejorar
    param_grid = {
        'fit_intercept': [True, False],
    }

    # cv=3 para un conjunto de validación del 1/3 de los datos de entrenamiento
    grid_search  = GridSearchCV(model, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error')
    # Entrenar el modelo con el conjunto de entrenamiento
    grid_search.fit(features_train, target_train)
    # Predicciones para el conjunto de validación
    predicts = grid_search.predict(features_valid)
    
    # Calculo del volumen promedio para las predicciones y el real
    predict_mean = predicts.mean()
    real_mean = target_valid.mean()
    # Imprimir las reservas previstas del conjunto de validación
    print(f'Volumen promedio de las reservas previstas: {predict_mean}')
    # Imprimir las reservas promedio del conjunto de validación
    print(f'Volumen promedio de las reservas reales: {real_mean}')
    
    # Calcular la raíz del error cuadrático medio
    rmse = mean_squared_error(predicts, target_valid) ** 0.5

    logging.info(f'RMSE: {rmse}')
    return predicts, grid_search.best_estimator_, rmse, predict_mean, real_mean

def scatter_predcts_target(predicts, target):
    ''' 
    Función para graficar las predicciones en una gráfica de disperción.
    Toma las predicciones y el objetivo.
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(target, predicts, alpha=0.5)  # Gráfico de dispersión
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Gráfico de dispersión: Valores reales vs Predicciones')
    # Línea diagonal (predicción perfecta)
    min_value = min(min(target), min(predicts))
    max_value = max(max(target), max(predicts))
    plt.plot([min_value, max_value], [min_value, max_value], 'k--')  # Línea diagonal
    
    plt.grid(True)
    plt.show()

def ganancia_predict(predicts, target):
    '''
    A apartir de las predicciones. Se toman los mejores 200 pozos y se obtiene el volumen
    original [miles de barriles] de esos pozos para calcular las ganancias.
    Las ganancias se calculan de la siguiente forma:
    volumen total x ingreso por unidad - presupuesto
    '''
    # Se inicializan las variables
    ingreso_por_unidad = 4500 # USD
    presupuesto = 100000000 # millones de dolares

    # Ordernar las predicciones de forma descendente y tomar las 200 mejores
    best_predicts = pd.Series(predicts).sort_values(ascending=False).head(200)
    
    # Resetear el index del objetivo de validacion
    target = target.reset_index(drop=True)
    
    # Obtener el volumen original de los pozos
    best_target = target[best_predicts.index]
    
    # Calcular la ganancia
    ganancia = best_target.sum() * ingreso_por_unidad - presupuesto
    
    return ganancia

def ganancia_predict(predicts, target):
    '''
    A apartir de las predicciones. Se toman los mejores 200 pozos y se obtiene el volumen
    original [miles de barriles] de esos pozos para calcular las ganancias.
    Las ganancias se calculan de la siguiente forma:
    volumen total x ingreso por unidad - presupuesto
    '''
     # Se inicializan las variables
    ingreso_por_unidad = 4500 # USD
    presupuesto = 100000000 # millones de dolares

    # Ordernar las predicciones de forma descendente y tomar las 200 mejores
    best_predicts = pd.Series(predicts).sort_values(ascending=False).head(200)
    
    # Resetear el index del objetivo de validacion
    target = target.reset_index(drop=True)
    
    # Obtener el volumen original de los pozos
    best_target = target[best_predicts.index]
    
    # Calcular la ganancia
    ganancia = best_target.sum() * ingreso_por_unidad - presupuesto
    
    return ganancia

def bootstrapping_ganancia(predicts, valid, n_muestras=1000):
    '''
    Se toman 1000 muestras con 500 pozos cada una. Se evalua la ganancia para
    el volumen original de los mejores 200 pozos. 
    '''
    # Crear una instancia para números aleatorios
    state = np.random.RandomState(54321)
    
    # Convertir a serie por comodidad
    predicts = pd.Series(predicts)
    
    # Bootstraping para 1000  muestras
    ganancias_muestras = []     # Guardar la ganancia calculada por muestra real
    for _ in range(n_muestras):
        # Tomar las muestras para las predicciones y el objetivo para n pozos
        predicts_subsample = predicts.sample(n=500, replace=True, random_state=state)
        # Calcular la ganancia
        ganancias_muestras.append(ganancia_predict(predicts_subsample, valid))

    return pd.Series(ganancias_muestras)