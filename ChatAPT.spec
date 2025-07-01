# ChatAPT.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
import os

block_cipher = None

# Helper para obtener la ruta base del proyecto (asumiendo que .spec está en la raíz)
project_root = os.path.abspath(os.path.dirname(__file__))

# Definir rutas a los scripts y directorios de datos
main_script = os.path.join(project_root, 'text_analysis', 'text_analysis.py')
checkpoints_dir_source = os.path.join(project_root, 'checkpoints')
checkpoints_dir_dest = 'checkpoints' # Relativo a la raíz del bundle
database_models_script_source = os.path.join(project_root, 'database_models.py')
database_models_script_dest = '.' # En la raíz del bundle
configure_db_script_source = os.path.join(project_root, 'configure_db.py') # No es parte de este bundle, pero para referencia
# config_ini_source = os.path.join(project_root, 'config.ini') # Si quieres empaquetar un config.ini por defecto
# config_ini_dest = '.'

# Lista de datos a incluir
datas_list = []
if os.path.exists(checkpoints_dir_source):
    datas_list.append((checkpoints_dir_source, checkpoints_dir_dest))
if os.path.exists(database_models_script_source):
    datas_list.append((database_models_script_source, database_models_script_dest))
# if os.path.exists(config_ini_source):
#    datas_list.append((config_ini_source, config_ini_dest))


a = Analysis([main_script],
             pathex=[project_root], # Añadir raíz del proyecto a pathex
             binaries=[],
             datas=datas_list,
             hiddenimports=[
                 'sqlalchemy.dialects.mssql',
                 'pyodbc',
                 'pandas._libs.tslibs', # Pandas a veces necesita esto explícitamente
                 'sklearn.utils._typedefs',
                 'sklearn.utils._heap',
                 'sklearn.utils._sorting',
                 'sklearn.neighbors._ball_tree', # Ejemplos comunes para sklearn
                 'sklearn.neighbors._kd_tree',
                 'sklearn.tree',
                 'sklearn.tree._utils',
                 'customtkinter',
                 'customtkinter.windows',
                 'customtkinter.widgets',
                 'customtkinter.draw_engine',
                 'customtkinter.appearance_mode_tracker',
                 'customtkinter.customtkinter_path',
                 'googletrans',
                 'wikipedia',
                 'duckduckgo_search',
                 'sentence_transformers',
                 'transformers', # sentence-transformers depende de transformers
                 'torch',
                 'joblib',
                 'configparser',
                 # Puede que necesites añadir más para torch, especialmente si usas MKL o backends específicos
                 # 'torch.backends.mkl'
             ],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='ChatAPT',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False, # True para debug, False para app windowed
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None)
          # icon=os.path.join(project_root, 'assets', 'app_icon.ico')) # Descomentar y ajustar ruta si tienes un icono
