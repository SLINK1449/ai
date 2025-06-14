import pyodbc
import pandas as pd

# connection configuration
server = 'localhost.localdomain'
database = 'TransformerNeuronDB'
username = 'sa'
password = ''

# optimize the connection string
conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password};'
   )

def load_data_from_sql(query):
    try:
        with pyodbc.connect(conn_str) as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error: {str(e)}")
        return pd.DataFrame()

# select model ai
data_frame = load_data_from_sql("SELECT TOP 10 * FROM ModelConfigs")

# show data
if not data_frame.empty:
    print("Datos recibidos correctamente:")
    print(data_frame.head())
else:
    print("No se encontraron datos.")
