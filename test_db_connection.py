import os
import sys
from sqlalchemy.orm import Session

# Asegurarse de que database_models.py sea importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # Asumiendo que database_models.py está en la raíz
from database_models import create_tables_if_not_exist, get_db_session

DB_CONFIG_TEST = {
    'server': os.environ.get('SQL_SERVER_NAME', 'localhost.localdomain'),
    'database': os.environ.get('SQL_DATABASE_NAME', 'TransformerNeuronDB'),
    'username': os.environ.get('SQL_USERNAME', 'sa'),
    'password': os.environ.get('SQL_PASSWORD', 'brian04271208@'),
    'driver': os.environ.get('SQL_DRIVER', 'ODBC Driver 17 for SQL Server').replace('+', ' ')
}

if __name__ == "__main__":
    print("Attempting to connect to DB and create tables...")
    try:
        # Primero, intentar crear/verificar las tablas
        create_tables_if_not_exist(DB_CONFIG_TEST)
        print("Table creation/check successful.")

        # Segundo, intentar obtener una sesión y cerrarla
        print("Attempting to get a DB session...")
        db_session = get_db_session(DB_CONFIG_TEST)
        if db_session:
            print("DB session obtained successfully.")
            # Realizar una consulta simple
            from sqlalchemy import text as sql_text # Importar text
            result = db_session.execute(sql_text("SELECT 1")).scalar_one()
            print(f"Simple query result: {result}")
            db_session.close()
            print("DB session closed.")
        else:
            print("Failed to obtain DB session.")
            sys.exit(1)

        print("Database connection test successful.")

    except ImportError as e:
        print(f"Import Error: {e}. Ensure database_models.py is in the correct path.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during DB connection test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
