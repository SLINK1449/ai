# ConfigureDB.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
import os

block_cipher = None

# Helper para obtener la ruta base del proyecto (asumiendo que .spec está en la raíz)
project_root = os.path.abspath(os.path.dirname(__file__))

# Definir rutas a los scripts y directorios de datos
main_script = os.path.join(project_root, 'configure_db.py')
database_models_script_source = os.path.join(project_root, 'database_models.py')
database_models_script_dest = '.' # En la raíz del bundle
# config_ini_source = os.path.join(project_root, 'config.ini') # config.ini es creado/leído, no necesariamente empaquetado como default
# config_ini_dest = '.'

# Lista de datos a incluir
datas_list = []
if os.path.exists(database_models_script_source):
    datas_list.append((database_models_script_source, database_models_script_dest))
# if os.path.exists(config_ini_source): # Descomentar si quieres un config.ini por defecto
#    datas_list.append((config_ini_source, config_ini_dest))


a = Analysis([main_script],
             pathex=[project_root],
             binaries=[],
             datas=datas_list,
             hiddenimports=[
                 'customtkinter',
                 'customtkinter.windows',
                 'customtkinter.widgets',
                 'customtkinter.draw_engine',
                 'customtkinter.appearance_mode_tracker',
                 'customtkinter.customtkinter_path',
                 'sqlalchemy.dialects.mssql',
                 'pyodbc',
                 'configparser'
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
          name='ConfigureDB',
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
          # icon=os.path.join(project_root, 'assets', 'config_icon.ico')) # Descomentar y ajustar ruta si tienes un icono
