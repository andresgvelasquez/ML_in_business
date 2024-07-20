# Importaciones para la gestion de credenciales
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import logging

import pandas as pd

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

    # Convertir los DataFrames a diccionarios
    logging.info(f'Resumen general: {general_summary_json}')
    logging.info(f'Resumen por columnas: {column_summary_json}')
    
    return general_summary_json , column_summary_json