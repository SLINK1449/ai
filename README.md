# Proyecto de IA con PyTorch: Aplicación de Escritorio (Chatbot y Clasificador)

Este proyecto contiene una aplicación de escritorio que integra dos funcionalidades principales de IA desarrolladas con PyTorch:

1.  **Chatbot Interactivo:**
    *   Responde preguntas utilizando información de una base de datos SQL Server, Wikipedia y DuckDuckGo.
    *   Utiliza `sentence-transformers` para la similitud semántica.
    *   Las interacciones se gestionan mediante SQLAlchemy con una base de datos SQL Server.
2.  **Inferencia con Clasificador Tabular:**
    *   Permite ingresar datos para dos características (`Feature1`, `Feature2`) y obtener una predicción de clase utilizando un modelo `TabularTransformer` preentrenado (cuantizado por defecto).

La aplicación de escritorio utiliza `CustomTkinter` para una interfaz gráfica moderna y está diseñada para ser compatible con CPU en Linux, Windows y macOS. El entrenamiento del modelo `TabularTransformer` se realiza a través de un script separado (`big_data_pro/datamodel.py`).

## 1. Requisitos Previos

*   **Python 3.8+**
*   **SQL Server:** Una instancia de SQL Server accesible (local o remota) para el funcionamiento del chatbot y para el entrenamiento del clasificador.
*   **Drivers ODBC para SQL Server:** Necesarios para que `pyodbc` (usado por SQLAlchemy) se conecte a SQL Server. La instalación varía según el sistema operativo.

## 2. Configuración del Entorno

Es altamente recomendable utilizar un entorno virtual de Python.

```bash
# Crear un entorno virtual (ejemplo con venv)
python3 -m venv ia_env

# Activar el entorno virtual
# Linux / macOS:
source ia_env/bin/activate
# Windows (PowerShell):
# .\ia_env\Scripts\Activate.ps1
# Windows (CMD):
# ia_env\Scripts\activate.bat
```

## 3. Instalación de Dependencias

Una vez activado el entorno virtual, instala las dependencias:

```bash
pip install -r requirements.txt
```
Esto incluye `PyTorch` (CPU), `SQLAlchemy`, `CustomTkinter`, `sentence-transformers`, `PyInstaller`, etc.

## 4. Configuración de la Conexión a SQL Server

La aplicación utiliza un archivo `config.ini` (en la raíz del proyecto o junto al ejecutable) para almacenar la configuración de la base de datos. Puedes gestionar esta configuración de tres maneras:

1.  **Usando la Herramienta de Configuración GUI (Recomendado):**
    Ejecuta la herramienta de configuración dedicada:
    ```bash
    python configure_db.py
    ```
    Esta GUI te permitirá ingresar los detalles de tu servidor SQL Server, probar la conexión y guardar la configuración en `config.ini`.

2.  **Mediante Variables de Entorno:**
    Los siguientes valores por defecto en `database_models.py` pueden ser sobrescritos por variables de entorno:
    *   `SQL_SERVER_NAME` (Default: `localhost.localdomain`)
    *   `SQL_DATABASE_NAME` (Default: `TransformerNeuronDB`)
    *   `SQL_USERNAME` (Default: `sa`)
    *   `SQL_PASSWORD` (Default: `brian04271208@`)
    *   `SQL_DRIVER` (Default: `ODBC Driver 17 for SQL Server`)
    Si `config.ini` existe, sus valores tendrán precedencia sobre las variables de entorno y los defaults internos.

3.  **Editando `config.ini` Manualmente:**
    Si `config.ini` se crea (por ejemplo, al ejecutar `configure_db.py` una vez), puedes editarlo directamente.

**Es crucial que el nombre del driver ODBC especificado coincida exactamente con el driver instalado en tu sistema.**

### Instalación de Drivers ODBC para SQL Server:

**(Las instrucciones para Linux, Windows y macOS permanecen igual que en la versión anterior del README - referirse a ellas si es necesario. Se omite aquí por brevedad, pero estarían presentes en el archivo real.)**

## 5. Creación del Esquema de la Base de Datos

Las tablas necesarias (`Documents`, `TrainingData`, `Predictions`) se definen como modelos ORM en `database_models.py`.

*   Al ejecutar `python configure_db.py` o `python text_analysis/text_analysis.py` por primera vez, se intentará crear el esquema en la base de datos si las tablas no existen.
*   **Para producción:** Se recomienda usar herramientas de migración como Alembic.

**Estructura de Datos para `TrainingData` (para el script de entrenamiento):**
*   `Feature1` (Float), `Feature2` (Float), `Target` (String), `SplitType` (String, e.g., 'Train').

## 6. Uso de la Aplicación

### 6.1. Configurar la Base de Datos (Primera Vez)
Antes de usar la aplicación principal, ejecuta la herramienta de configuración:
```bash
python configure_db.py
```
Ingresa los detalles de tu SQL Server, prueba la conexión y guarda.

### 6.2. Ejecutar la Aplicación Principal (Chatbot y Clasificador)
Una vez configurada la base de datos:
```bash
python text_analysis/text_analysis.py
```
Esto abrirá la aplicación de escritorio con dos pestañas:
*   **Chatbot:** Interactúa con el chatbot.
*   **Tabular Classifier:** Ingresa valores para `Feature1` y `Feature2` y obtén una predicción usando el modelo preentrenado. El modelo, scaler y label encoder se cargan desde el directorio `checkpoints/`.

### 6.3. Entrenar el Modelo Clasificador Tabular (Opcional, Avanzado)
Si deseas re-entrenar el `TabularTransformer` (por ejemplo, con tus propios datos en la tabla `TrainingData`):
```bash
python big_data_pro/datamodel.py
```
Esto generará/actualizará los archivos en `checkpoints/` (`.pth` para modelos, `.joblib` para scaler/encoder) que usa la pestaña "Tabular Classifier" en la GUI principal.

## 7. Empaquetado como Aplicación de Escritorio (Standalone)

Este proyecto incluye archivos `.spec` para PyInstaller, permitiendo crear ejecutables standalone.

1.  **Asegúrate de tener PyInstaller instalado** (está en `requirements.txt`).
2.  **Navega a la raíz del proyecto** en tu terminal.
3.  **Construir la aplicación principal:**
    ```bash
    pyinstaller ChatAPT.spec
    ```
4.  **Construir la herramienta de configuración:**
    ```bash
    pyinstaller ConfigureDB.spec
    ```
5.  Los ejecutables se encontrarán en las carpetas `dist/ChatAPT` y `dist/ConfigureDB` respectivamente.

**Notas sobre el Empaquetado:**
*   Los archivos `.spec` están configurados para incluir los archivos necesarios (como `database_models.py` y el directorio `checkpoints`).
*   **`config.ini`:** El archivo `config.ini` se leerá desde la misma ubicación que el ejecutable. Al ejecutar por primera vez la aplicación empaquetada, es posible que necesites ejecutar primero `ConfigureDB` (empaquetado) para crear/ajustar `config.ini` en esa ubicación.
*   **Modelos de `sentence-transformers`:** Estos modelos se descargan y cachean la primera vez que se usan. Para una distribución offline completa, necesitarías empaquetar estos modelos cacheados. Esto es un paso avanzado y no está cubierto por el `.spec` actual por defecto.
*   **Depuración de PyInstaller:** Si el empaquetado falla o el ejecutable no funciona, revisa los logs. Puede que necesites ajustar la sección `hiddenimports` en los archivos `.spec` o añadir "hooks" para librerías complejas.

## 8. Optimizaciones Implementadas

*   **GUI Moderna:** Uso de `CustomTkinter`.
*   **PyTorch Threads, `torch.compile`, Cuantización (para el clasificador).**
*   **Paralelización I/O y Caching (para el chatbot).**
*   **Configuración de BD Centralizada y Editable.**

## 9. Estructura del Proyecto (Resumen)

*   `text_analysis/text_analysis.py`: Script GUI principal (Chatbot + Inferencia Clasificador).
*   `big_data_pro/datamodel.py`: Script para entrenamiento del clasificador tabular.
*   `configure_db.py`: Script GUI para configurar la conexión a la BD.
*   `database_models.py`: Modelos ORM, lógica de `config.ini`, `DB_CONFIG` central.
*   `checkpoints/`: Modelos entrenados, scalers, encoders.
*   `config.ini`: (Creado por `configure_db.py` o manualmente) Almacena la configuración de la BD.
*   `ChatAPT.spec`, `ConfigureDB.spec`: Archivos de configuración de PyInstaller.
*   `requirements.txt`, `README.md`.

## 10. Solución de Problemas Comunes
**(Similar a la sección anterior, enfatizando la configuración del driver ODBC y la ubicación de `config.ini` para la app empaquetada).**

---
Disfruta usando y experimentando con estas aplicaciones de IA!
