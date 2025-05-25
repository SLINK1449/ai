"""
Transformer-based Classifier with SQL Server Integration
Author: Your Name
Date: YYYY-MM-DD
Description: End-to-end pipeline for training Transformer models with SQL data.
"""
import time
import torch
import torch.nn as nn
import pandas as pd
import pyodbc
import numpy as np
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# cronómetro global
start_time = time.time()

# ==================================================================
# Database Configuration (secure)
# ==================================================================
DB_CONFIG = {
    'server': 'localhost.localdomain',
    'database': 'TransformerNeuronDB',
    'username': 'sa',
    'password': '',
    'driver': '{ODBC Driver 17 for SQL Server}'
}

# ==================================================================
# SQL Queries 
# ==================================================================
TRAINING_QUERY = """
SELECT Feature1, Feature2, Target 
FROM [dbo].[TrainingData]
WHERE SplitType = 'Train'
"""

PREDICTIONS_INSERT_QUERY = """
INSERT INTO Predictions (Feature1, Feature2, Prediction, Confidence) 
VALUES (?, ?, ?, ?)
"""

# ==================================================================
# Database Helper Functions
# ==================================================================
def get_db_connection():
    """Create and return a database connection."""
    conn_str = (
        f'DRIVER={DB_CONFIG["driver"]};'
        f'SERVER={DB_CONFIG["server"]};'
        f'DATABASE={DB_CONFIG["database"]};'
        f'UID={DB_CONFIG["username"]};'
        f'PWD={DB_CONFIG["password"]};'
        'Encrypt=no;'
        'TrustServerCertificate=yes;'
    )
    return pyodbc.connect(conn_str)

def execute_sql_query(query, params=None, fetch=True):
    """Execute a SQL query and return results."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        if fetch:
            columns = [column[0] for column in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame.from_records(data, columns=columns)
        else:
            conn.commit()
            return None

# ==================================================================
# Model Definition
# ==================================================================
class TabularTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.fc(x)

# ==================================================================
# Training Pipeline
# ==================================================================
def train_model(pipe):
    """Main training pipeline for the Transformer model."""
    print("Loading training data from SQL...")
    train_df = execute_sql_query(TRAINING_QUERY)
    
    X = train_df[['Feature1', 'Feature2']].values
    y = train_df['Target'].values
    unique_classes = np.unique(y)
    print(f"Detected classes in Target: {unique_classes}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    model = TabularTransformer(input_dim=X_train_tensor.shape[1], d_model=64, nhead=4, num_layers=2, num_classes=len(unique_classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting infinite training (press Ctrl+C to stop)...")
    epoch = 1
    try:
        while True:
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            print(f"Epoch {epoch:05} | Loss: {loss_value:.6f}")

            # Enviar datos a la interfaz gráfica si el pipe está activo
            if pipe:
                try:
                    pipe.send({'epoch': epoch, 'loss': loss_value})
                except:
                    pass

            epoch += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Proceeding to validation and saving...")

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_preds = torch.argmax(val_outputs, dim=1)
        confidences = torch.softmax(val_outputs, dim=1).max(dim=1).values
        accuracy = (val_preds == y_val_tensor).float().mean()

    print(f"\nValidation Accuracy: {accuracy.item()*100:.2f}%")
    print(classification_report(y_val_tensor.numpy(), val_preds.numpy()))

    print("\nSaving predictions to database...")
    predictions_df = pd.DataFrame({
        'Feature1': X_val[:, 0],
        'Feature2': X_val[:, 1],
        'Prediction': val_preds.numpy(),
        'Confidence': confidences.numpy()
    })

    with get_db_connection() as conn:
        cursor = conn.cursor()
        for _, row in predictions_df.iterrows():
            cursor.execute(PREDICTIONS_INSERT_QUERY, (
                row['Feature1'], 
                row['Feature2'], 
                int(row['Prediction']), 
                float(row['Confidence'])
            ))
        conn.commit()

    print(f"Successfully saved {len(predictions_df)} predictions!")
    torch.save(model.state_dict(), 'transformer_model.pth')

# ==================================================================
# Main Execution
# ==================================================================
if __name__ == "__main__":
    try:
        train_model()
    except pyodbc.InterfaceError as e:
        print("\n[ERROR] Database connection failed:", e)
    elapsed_time = time.time() - start_time
    print(f"\nPipeline execution completed!")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
