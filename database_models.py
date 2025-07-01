from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine.url import URL

# Define la base para los modelos declarativos
Base = declarative_base()

# --- Modelos ORM ---

class Document(Base):
    """
    ORM model for the 'Documents' table.
    Stores text data, user questions, and sources from text_analysis.
    """
    __tablename__ = 'Documents'

    Id = Column(Integer, primary_key=True, autoincrement=True)
    UserQuestion = Column(Text, nullable=True) # Assuming UserQuestion can sometimes be null
    Description = Column(Text, nullable=False)
    Source = Column(String(255), nullable=True)
    SearchPattern = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Document(Id={self.Id}, Source='{self.Source}', Question='{self.UserQuestion[:20]}...')>"

class TrainingData(Base):
    """
    ORM model for the 'TrainingData' table.
    Stores features and target for training the TabularTransformer model.
    """
    __tablename__ = 'TrainingData'

    Id = Column(Integer, primary_key=True, autoincrement=True) # Assuming an ID is needed
    Feature1 = Column(Float, nullable=False)
    Feature2 = Column(Float, nullable=False)
    Target = Column(String(255), nullable=False) # Assuming Target is a string label
    SplitType = Column(String(50), nullable=True, default='Train') # e.g., 'Train', 'Test', 'Validate'

    def __repr__(self):
        return f"<TrainingData(Id={self.Id}, Target='{self.Target}', Split='{self.SplitType}')>"

class Prediction(Base):
    """
    ORM model for the 'Predictions' table.
    Stores predictions made by the TabularTransformer model.
    """
    __tablename__ = 'Predictions'

    Id = Column(Integer, primary_key=True, autoincrement=True)
    Feature1 = Column(Float, nullable=False)
    Feature2 = Column(Float, nullable=False)
    Prediction = Column(Integer, nullable=False) # Storing the encoded prediction label
    Confidence = Column(Float, nullable=False)

    # Optional: If predictions are linked to specific training data or documents
    # TrainingDataId = Column(Integer, ForeignKey('TrainingData.Id'))
    # DocumentId = Column(Integer, ForeignKey('Documents.Id'))

    def __repr__(self):
        return f"<Prediction(Id={self.Id}, PredictedLabel={self.Prediction}, Confidence={self.Confidence:.2f})>"


# --- Configuración de la Base de Datos y Sesión (Centralizada con config.ini) ---
import os
import sys # For sys._MEIPASS
import configparser
from sqlalchemy.exc import OperationalError

# --- Utility for Resource Path (for PyInstaller) ---
def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not running in a PyInstaller bundle, use script's directory
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        # If database_models.py is at root, and config.ini is also at root.
        # If script is in a subdir, adjust relative_path or base_path accordingly.
        # This assumes config.ini is at the same level as where this function is called from,
        # or relative_path correctly navigates from the executable's location.
        # For config.ini at project root, when this file is also at root:
        # Pass "config.ini" as relative_path.

    return os.path.join(base_path, relative_path)

CONFIG_FILE_PATH = get_resource_path("config.ini") # Use the helper
DB_SECTION = "Database"

# Valores por defecto si config.ini o variables de entorno no existen
DEFAULT_DB_CONFIG = {
    'server': 'localhost.localdomain',
    'database': 'TransformerNeuronDB',
    'username': 'sa',
    'password': 'brian04271208@',
    'driver': 'ODBC Driver 17 for SQL Server' # Stored with spaces, URL encoding handled later
}

def load_db_config() -> dict:
    """
    Loads database configuration with priority:
    1. config.ini file
    2. Environment variables
    3. Default values
    """
    config = configparser.ConfigParser()
    loaded_config = DEFAULT_DB_CONFIG.copy() # Start with defaults

    # Try environment variables first (as they might be more dynamic/secure)
    env_server = os.environ.get('SQL_SERVER_NAME')
    env_database = os.environ.get('SQL_DATABASE_NAME')
    env_username = os.environ.get('SQL_USERNAME')
    env_password = os.environ.get('SQL_PASSWORD')
    env_driver = os.environ.get('SQL_DRIVER')

    if env_server: loaded_config['server'] = env_server
    if env_database: loaded_config['database'] = env_database
    if env_username: loaded_config['username'] = env_username
    if env_password: loaded_config['password'] = env_password
    if env_driver: loaded_config['driver'] = env_driver.replace('+', ' ') # Store with spaces

    # Then, try to load from config.ini, potentially overriding env vars if file is more specific
    if os.path.exists(CONFIG_FILE_PATH): # Use CONFIG_FILE_PATH
        config.read(CONFIG_FILE_PATH)    # Use CONFIG_FILE_PATH
        if DB_SECTION in config:
            for key in loaded_config.keys():
                if key in config[DB_SECTION] and config[DB_SECTION][key]: # Ensure value is not empty
                    loaded_config[key] = config[DB_SECTION][key]
    else:
        print(f"[INFO] {CONFIG_FILE_PATH} not found. Using defaults or environment variables for DB config.")
        # Optionally, create a default config.ini here if it's the first run and desired
        # save_db_config(loaded_config) # Example: save current config if file missing

    return loaded_config

def save_db_config(config_dict: dict):
    """Saves the given database configuration to config.ini."""
    config = configparser.ConfigParser()
    config[DB_SECTION] = config_dict
    try:
        with open(CONFIG_FILE_PATH, 'w') as configfile: # Use CONFIG_FILE_PATH
            config.write(configfile)
        print(f"Database configuration saved to {CONFIG_FILE_PATH}")
        # Update in-memory DB_CONFIG and reinitialize engine
        global DB_CONFIG, engine, SessionLocal, SQLALCHEMY_DATABASE_URL
        DB_CONFIG = config_dict.copy() # Update global
        SQLALCHEMY_DATABASE_URL = _build_sqlalchemy_url(DB_CONFIG)
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        print("SQLAlchemy engine reinitialized with new configuration.")

    except IOError as e:
        print(f"[ERROR] Could not write {CONFIG_FILE_NAME}: {e}")


# --- Global DB variables, initialized by load_db_config() ---
DB_CONFIG = load_db_config()

def _build_sqlalchemy_url(current_db_config: dict) -> URL:
    """Helper to build SQLAlchemy URL from a config dict."""
    return URL.create(
        drivername='mssql+pyodbc',
        username=current_db_config.get('username'),
        password=current_db_config.get('password'),
        host=current_db_config.get('server'),
        database=current_db_config.get('database'),
        query={'driver': current_db_config.get('driver', '').replace(' ', '+')}
    )

SQLALCHEMY_DATABASE_URL = _build_sqlalchemy_url(DB_CONFIG)
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    """
    Creates and returns a new SQLAlchemy session using the (potentially updated) engine.
    """
    return SessionLocal()

def test_db_connection(test_config: dict = None) -> bool:
    """Tests database connection with the given or current config."""
    effective_config = test_config if test_config else DB_CONFIG
    temp_url = _build_sqlalchemy_url(effective_config)
    temp_engine = create_engine(temp_url)
    try:
        with temp_engine.connect() as connection:
            print(f"Successfully connected to DB with config: {effective_config['server']}/{effective_config['database']}")
            return True
    except OperationalError as e:
        print(f"DB Connection Test Failed for {effective_config['server']}/{effective_config['database']}: {e}")
        return False
    except Exception as e: # Catch other potential errors like driver issues
        print(f"An unexpected error occurred during DB connection test: {e}")
        return False


def create_tables_if_not_exist():
    """
    Creates all tables defined in Base.metadata if they do not already exist,
    using the centrally defined engine.
    """
    global engine # Ensure we use the potentially re-initialized engine
    try:
        Base.metadata.create_all(bind=engine)
        print("Tables checked/created based on ORM models using central configuration.")
    except Exception as e:
        print(f"Error creating tables: {e}")
        print("Please ensure the database server is running and accessible, and credentials are correct.")
        # Do not raise here, allow app to start and user to fix config via UI
        # raise


# Si este archivo se ejecuta directamente, podría ser para crear las tablas o testear config.
if __name__ == '__main__':
    print("database_models.py executed.")
    print(f"Current DB_CONFIG loaded: {DB_CONFIG}")
    print(f"SQLAlchemy Database URL: {SQLALCHEMY_DATABASE_URL}")

    print("\nAttempting to test DB connection with current config...")
    if test_db_connection():
        print("Attempting to create tables (if they don't exist)...")
        try:
            create_tables_if_not_exist()
        except Exception as e:
            print(f"Failed to create tables directly: {e}")
    else:
        print("Skipping table creation due to connection test failure.")

    # Example of saving a slightly modified config (for testing save_db_config)
    # current_config = load_db_config()
    # current_config['username'] = 'test_user_save'
    # save_db_config(current_config)
    # print(f"New DB_CONFIG after save: {DB_CONFIG}")
    # test_db_connection() # Test again with new config
    pass
