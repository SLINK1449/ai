import customtkinter as ctk
import tkinter.messagebox as messagebox
import os
import sys

# Asegurarse de que database_models.py sea importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from database_models import DB_CONFIG, save_db_config, test_db_connection, CONFIG_FILE_NAME, load_db_config

class DBConfigApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Database Configuration Tool")
        self.geometry("550x450") # Ajustado para más campos y mensajes
        ctk.set_appearance_mode("System") # System, Dark, Light
        ctk.set_default_color_theme("blue") # blue, green, dark-blue

        self.config_vars = {}
        self.config_entries = {}

        # Frame principal
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        main_frame.grid_columnconfigure(1, weight=1) # Columna de Entries se expande

        # Título
        title_label = ctk.CTkLabel(main_frame, text="Database Connection Settings", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Campos de configuración
        config_fields = [
            ("Server Host", "server"),
            ("Database Name", "database"),
            ("Username", "username"),
            ("Password", "password"),
            ("ODBC Driver Name", "driver")
        ]

        # Cargar configuración actual para poblar los campos
        current_config = load_db_config() # Usar la función que ya tiene la lógica de carga

        for i, (label_text, key) in enumerate(config_fields):
            label = ctk.CTkLabel(main_frame, text=f"{label_text}:")
            label.grid(row=i + 1, column=0, padx=(0,10), pady=5, sticky="w")

            self.config_vars[key] = ctk.StringVar(value=current_config.get(key, ""))
            entry_widget = ctk.CTkEntry(main_frame, textvariable=self.config_vars[key], width=300)
            if key == "password":
                entry_widget.configure(show="*")
            entry_widget.grid(row=i + 1, column=1, padx=5, pady=5, sticky="ew")
            self.config_entries[key] = entry_widget

        # Frame para botones
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.grid(row=len(config_fields) + 1, column=0, columnspan=2, pady=(20, 10))
        button_frame.grid_columnconfigure((0,1), weight=1)


        self.save_button = ctk.CTkButton(button_frame, text="Save Configuration", command=self._save_config)
        self.save_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.test_button = ctk.CTkButton(button_frame, text="Test Connection", command=self._test_connection)
        self.test_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Etiqueta de estado
        self.status_label = ctk.CTkLabel(main_frame, text="", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=len(config_fields) + 2, column=0, columnspan=2, pady=(10,0))

        config_path_label = ctk.CTkLabel(main_frame, text=f"Configuration will be saved to: {os.path.abspath(CONFIG_FILE_NAME)}", font=ctk.CTkFont(size=10), text_color="gray")
        config_path_label.grid(row=len(config_fields) + 3, column=0, columnspan=2, pady=(5,0))


    def _get_current_form_config(self) -> dict:
        """Lee los valores actuales de los campos de entrada."""
        form_config = {}
        for key, var in self.config_vars.items():
            form_config[key] = var.get()
        return form_config

    def _save_config(self):
        self.status_label.configure(text="") # Limpiar estado
        current_form_config = self._get_current_form_config()

        # Validar que los campos no estén vacíos (opcional, pero bueno)
        for key, value in current_form_config.items():
            if not value:
                self.status_label.configure(text=f"Error: '{key}' cannot be empty.", text_color="red")
                messagebox.showerror("Error", f"Field '{key}' cannot be empty.")
                return

        try:
            save_db_config(current_form_config) # Esta función está en database_models.py
            self.status_label.configure(text=f"Configuration saved to {CONFIG_FILE_NAME}", text_color="green")
            messagebox.showinfo("Success", f"Configuration saved successfully to {CONFIG_FILE_NAME}.\nThe main application will use this new configuration on its next run (or if it re-initializes its DB connection).")
        except Exception as e:
            self.status_label.configure(text=f"Error saving config: {e}", text_color="red")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def _test_connection(self):
        self.status_label.configure(text="Testing connection...", text_color="gray")
        current_form_config = self._get_current_form_config()

        # Validar que los campos no estén vacíos antes de probar
        for key, value in current_form_config.items():
            if not value:
                self.status_label.configure(text=f"Error: '{key}' cannot be empty for testing.", text_color="red")
                messagebox.showerror("Error", f"Field '{key}' cannot be empty for connection testing.")
                return

        if test_db_connection(current_form_config): # test_db_connection está en database_models.py
            self.status_label.configure(text="Connection successful!", text_color="green")
            messagebox.showinfo("Connection Test", "Successfully connected to the database with the provided settings.")
        else:
            self.status_label.configure(text="Connection failed. Check settings and DB server.", text_color="red")
            messagebox.showerror("Connection Test", "Failed to connect to the database. Please check your settings, ODBC driver, and ensure the SQL Server is accessible.")

if __name__ == "__main__":
    app = DBConfigApp()
    app.mainloop()
