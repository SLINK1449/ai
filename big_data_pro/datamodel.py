"""
Transformer-based Classifier for Tabular Data with SQL Server Integration.

This module defines and trains a Transformer model for classification tasks on
tabular data sourced from a SQL Server database. It includes data loading,
preprocessing, model definition, training, evaluation, and prediction saving.
"""
import time
import os
import platform # For OS specific details if needed & thread info
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sqlalchemy import select, text as sql_text # For SQLAlchemy select and text queries
from sqlalchemy.orm import Session # For type hinting

import pandas as pd
# import pyodbc # No longer directly needed here, SQLAlchemy will handle it
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch.quantization # For quantization
import joblib # For saving/loading scaler and label_encoder

# --- Configuration ---
# Consider moving to a separate config file or using environment variables for sensitive info
# This DB_CONFIG should align with what database_models.get_db_session expects
# DB_CONFIG is now imported from database_models
# Import ORM models and session utility, and resource path helper
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database_models import DB_CONFIG, TrainingData, Prediction, get_db_session, create_tables_if_not_exist, get_resource_path

# Context manager for database sessions
from contextlib import contextmanager
@contextmanager
def get_db_session_context():
    """Provides a SQLAlchemy session that is automatically closed."""
    db = None
    try:
        db = get_db_session() # Uses centralized DB_CONFIG from database_models
        yield db
    finally:
        if db:
            db.close()

# Model Hyperparameters (can be tuned)
INPUT_DIM = 2  # Number of features (Feature1, Feature2) # This might need to be dynamic if features change
D_MODEL = 64   # Embedding dimension for the transformer
N_HEAD = 4     # Number of attention heads
NUM_LAYERS = 2 # Number of transformer encoder layers
# NUM_CLASSES will be determined from data
DROPOUT = 0.1  # Dropout rate

# Training Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32 # Adjusted for potentially small datasets from SQL
EPOCHS = 100 # Number of epochs for training (finite loop)
WEIGHT_DECAY = 1e-4 # For AdamW optimizer

MODEL_SAVE_PATH = "../transformer_model.pth" # Save in parent directory from big_data_pro
# Ensure checkpoints directory exists if used for more granular saving
# Use get_resource_path to make CHECKPOINT_DIR relative to the application root or executable location
# If database_models.py (where get_resource_path is) is at root, and checkpoints is also at root.
# This assumes datamodel.py is in a subdirectory like big_data_pro/.
# So, get_resource_path('../checkpoints') would navigate up one level then into checkpoints.
# Or, if get_resource_path is designed to work from script location:
# base_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# CHECKPOINT_DIR = os.path.join(base_app_dir, "checkpoints")
# A simpler way if get_resource_path assumes it's called from a script at root or gives path relative to executable:
CHECKPOINT_DIR_NAME = "checkpoints"
CHECKPOINT_DIR = get_resource_path(CHECKPOINT_DIR_NAME) # This will place checkpoints next to executable/main script

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "tabular_transformer_final_fp32.pth")
BEST_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "tabular_transformer_best_fp32.pth")
QUANTIZED_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "tabular_transformer_quantized_int8.pth")
SCALER_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "tabular_scaler.joblib")
LABEL_ENCODER_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "tabular_label_encoder.joblib")


# --- SQL Queries ---
# It's good practice to define table and column names as constants if they are used in many places
TABLE_TRAINING_DATA = "[dbo].[TrainingData]"
COLUMN_FEATURE1 = "Feature1"
COLUMN_FEATURE2 = "Feature2"
COLUMN_TARGET = "Target"
COLUMN_SPLIT_TYPE = "SplitType" # Assuming a column to differentiate train/test/val data in SQL

TRAINING_DATA_QUERY = f"""
SELECT {COLUMN_FEATURE1}, {COLUMN_FEATURE2}, {COLUMN_TARGET}
FROM {TABLE_TRAINING_DATA}
WHERE {COLUMN_SPLIT_TYPE} = 'Train'
"""
# Add similar queries for validation/test data if they exist in SQL with different SplitType
# VALIDATION_DATA_QUERY = "..."

# The PREDICTIONS_INSERT_QUERY is no longer needed as we'll use ORM objects.

# --- Database Utilities (SQLAlchemy based) ---
# Old get_db_connection and execute_sql_query are removed.
# We now use get_db_session_context from database_models.py (via context manager above)

# --- Model Definition ---
class TabularTransformer(nn.Module):
    """
    A Transformer model adapted for tabular data classification.
    It uses a linear embedding for input features, followed by a standard
    Transformer Encoder, and a final fully connected layer for classification.
    """
    def __init__(self, input_dim: int, d_model: int, n_head: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        # TransformerEncoderLayer requires d_model, nhead
        # batch_first=True means input tensors will have batch size as the first dimension
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # x shape: (batch_size, input_dim)
        x_embedded = self.embedding(x)
        # x_embedded shape: (batch_size, d_model)

        # TransformerEncoder expects input of shape (batch_size, sequence_length, d_model)
        # For tabular data, we can treat each sample as a sequence of length 1.
        x_seq = x_embedded.unsqueeze(1)
        # x_seq shape: (batch_size, 1, d_model)

        transformer_out = self.transformer_encoder(x_seq)
        # transformer_out shape: (batch_size, 1, d_model)

        # We take the output corresponding to the single item in the sequence
        transformer_out_condensed = transformer_out.squeeze(1)
        # transformer_out_condensed shape: (batch_size, d_model)

        logits = self.output_fc(transformer_out_condensed)
        # logits shape: (batch_size, num_classes)
        return logits

# --- Data Handling and Preprocessing ---
def load_and_preprocess_data(query: str, target_column: str):
    """
    Loads data using the given SQL query, preprocesses it for the model.
    This includes scaling numerical features and encoding the target variable.
    Args:
        query (str): SQL query to fetch training data.
        target_column (str): Name of the target variable column.
    Returns:
        Tuple: (X_scaled_tensor, y_encoded_tensor, scaler, label_encoder, num_classes)
    """
    print("Loading training data from SQL...")
    df = execute_sql_query(query)
    if df is None or df.empty:
        print("[ERROR] No data loaded. Please check SQL query and database content.")
        return None, None, None, None, 0

    # Assuming features are all columns except the target
    # TODO: Make feature selection more robust if there are other non-feature columns
    feature_columns = [col for col in df.columns if col != target_column]
    if not feature_columns:
        print(f"[ERROR] No feature columns found. Target column: {target_column}, DataFrame columns: {df.columns.tolist()}")
        return None, None, None, None, 0

    X = df[feature_columns].values
    y = df[target_column].values

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Detected classes in '{target_column}': {label_encoder.classes_} (Encoded: {np.unique(y_encoded)})")
    print(f"Number of classes: {num_classes}")


    X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_encoded_tensor = torch.tensor(y_encoded, dtype=torch.long) # CrossEntropyLoss expects long type for targets

    return X_scaled_tensor, y_encoded_tensor, scaler, label_encoder, num_classes


def load_training_data_orm(target_column_name: str = COLUMN_TARGET, split_type_filter: str = 'Train'):
    """
    Loads training data from the database using SQLAlchemy ORM and returns a Pandas DataFrame.
    Args:
        target_column_name (str): The name of the target column.
        split_type_filter (str): The value for SplitType to filter training data (e.g., 'Train').
    Returns:
        pd.DataFrame: DataFrame containing features and target, or empty DataFrame on error.
    """
    print(f"Loading training data from SQL using ORM (SplitType='{split_type_filter}')...")
    records = []
    try:
        with get_db_session_context() as db: # Use the context manager
            # Select the entire TrainingData object/entity
            stmt = select(TrainingData).where(TrainingData.SplitType == split_type_filter)
            results = db.execute(stmt).scalars().all() # Gets a list of TrainingData objects

            if not results:
                print(f"[WARN] No training data found for SplitType='{split_type_filter}'.")
                return pd.DataFrame()

            # Convert list of ORM objects to a list of dictionaries for DataFrame creation
            for record_obj in results:
                records.append({
                    COLUMN_FEATURE1: record_obj.Feature1,
                    COLUMN_FEATURE2: record_obj.Feature2,
                    target_column_name: record_obj.Target
                    # Add other relevant columns if needed, ensuring they exist in TrainingData model
                })

            df = pd.DataFrame(records)

        if df.empty and results: # Should not happen if results were processed
             print(f"[WARN] Data was fetched but DataFrame is empty. Check record processing.")
        elif df.empty:
             print(f"[INFO] No training data resulted in an empty DataFrame (SplitType='{split_type_filter}').")

        return df
    except Exception as e: # Catch a broader range of exceptions
        print(f"[ERROR] ORM Data Loading (objects to DataFrame): {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame() # Return empty DataFrame on error


# --- Data Handling and Preprocessing (Modified to use load_training_data_orm) ---
def load_and_preprocess_data_orm(target_column: str = COLUMN_TARGET, split_filter: str = 'Train'):
    """
    Loads data using ORM (fetching ORM objects), preprocesses it for the model (scaling, encoding).
    """
    df = load_training_data_orm(target_column_name=target_column, split_type_filter=split_filter)

    if df is None or df.empty: # df could be None if load_training_data_orm has an unhandled case, though it returns empty df.
        print("[ERROR] No data loaded via ORM or DataFrame is None. Please check database content and ORM loading logic.")
        return None, None, None, None, 0

    feature_columns = [col for col in df.columns if col != target_column]
    if not feature_columns: # Should be Feature1, Feature2
        print(f"[ERROR] No feature columns found in DataFrame from ORM. Columns: {df.columns.tolist()}")
        return None, None, None, None, 0

    X = df[feature_columns].values
    y = df[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Detected classes in '{target_column}': {label_encoder.classes_} (Encoded: {np.unique(y_encoded)})")
    print(f"Number of classes: {num_classes}")

    X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_encoded_tensor = torch.tensor(y_encoded, dtype=torch.long)

    return X_scaled_tensor, y_encoded_tensor, scaler, label_encoder, num_classes


# --- Training and Evaluation ---
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, label_encoder: LabelEncoder):
    """Evaluates the model on the given dataloader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_confidences = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    # Ensure all_targets and all_preds are not empty before calculating accuracy
    if not all_targets or not all_preds:
        print("[WARN] Evaluation dataloader was empty or produced no predictions.")
        return avg_loss, 0.0, [], [], []

    accuracy = accuracy_score(all_targets, all_preds)
    # print(classification_report(all_targets, all_preds, target_names=label_encoder.classes_, zero_division=0))
    return avg_loss, accuracy, all_preds, all_targets, all_confidences


# --- Quantization Helper ---
def calibrate_model_for_quantization(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Calibrates the model with data from the dataloader.
    This is a necessary step for post-training static quantization.
    """
    model.eval() # Model must be in eval mode for calibration
    with torch.no_grad():
        for batch_X, _ in dataloader: # Only need inputs for calibration
            batch_X = batch_X.to(device)
            _ = model(batch_X) # Pass data through the model
    print("Model calibration complete.")

def quantize_and_save_model(
    original_model: nn.Module,
    calibration_dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    input_dim: int,
    d_model: int,
    n_head: int,
    num_layers: int,
    num_classes: int,
    dropout_rate: float
):
    """
    Performs post-training static quantization on the model and saves it.
    """
    print(f"\nStarting quantization process for model to be saved at {save_path}...")

    # 1. Create a new instance of the model for quantization to avoid issues with compiled models or state
    # Ensure this new model is on the CPU as quantization is typically CPU-focused.
    model_to_quantize = TabularTransformer(
        input_dim=input_dim, d_model=d_model, n_head=n_head,
        num_layers=num_layers, num_classes=num_classes, dropout=dropout_rate
    ).to(torch.device("cpu")) # Move to CPU

    # Load the state_dict from the best trained FP32 model
    # Ensure the original_model's state_dict is loaded from a CPU map_location if it was on GPU
    # If original_model is already a loaded state_dict (from BEST_MODEL_SAVE_PATH), this step might vary.
    # Assuming original_model is the nn.Module object that was trained.
    # If original_model is compiled, use _orig_mod if available, or load from BEST_MODEL_SAVE_PATH
    if hasattr(original_model, '_orig_mod'):
        model_to_quantize.load_state_dict(original_model._orig_mod.state_dict())
    else:
        model_to_quantize.load_state_dict(original_model.state_dict())

    model_to_quantize.eval() # Set to eval mode

    # 2. Specify quantization configuration
    # For typical server-side inference on x86, 'fbgemm' is a common backend.
    # For ARM, 'qnnpack' might be used. Ensure PyTorch is built with the chosen backend.
    quant_config = torch.quantization.get_default_qconfig('fbgemm') # For x86
    # Or, more generally for server: torch.quantization.default_qconfig
    # Or for mobile: torch.quantization.default_mobile_qconfig

    model_to_quantize.qconfig = quant_config
    print(f"Quantization config set to: {model_to_quantize.qconfig}")

    # 3. Prepare the model for static quantization. This inserts observer modules.
    torch.quantization.prepare(model_to_quantize, inplace=True)
    print("Model prepared for quantization (observers inserted).")

    # 4. Calibrate the model with representative data
    print("Calibrating model with data...")
    calibrate_model_for_quantization(model_to_quantize, calibration_dataloader, torch.device("cpu")) # Calibration on CPU

    # 5. Convert the model to a quantized version (INT8)
    quantized_model = torch.quantization.convert(model_to_quantize, inplace=True)
    print("Model converted to quantized version (INT8).")

    # 6. Save the quantized model
    # Use torch.jit.script for better portability if needed, or just save state_dict
    # For state_dict of a quantized model, it's usually just saved like a regular model.
    # However, for easier deployment, scripting is often preferred.
    try:
        # Scripting can sometimes fail with complex model structures or certain ops after quantization.
        scripted_quantized_model = torch.jit.script(quantized_model)
        torch.jit.save(scripted_quantized_model, save_path)
        print(f"Scripted quantized model saved to {save_path}")
    except Exception as e:
        print(f"Failed to script quantized model: {e}. Saving state_dict instead.")
        torch.save(quantized_model.state_dict(), save_path + ".state_dict") # Fallback
        print(f"Quantized model state_dict saved to {save_path}.state_dict")

    return quantized_model


def training_pipeline(num_epochs: int,
                      batch_size: int,
                      learning_rate: float,
                      weight_decay: float,
                      pipe_to_ui=None): # Optional pipe for UI communication
    """
    Main training pipeline for the TabularTransformer model.
    Includes data loading, model initialization, training loop, evaluation, and saving.
    """
    start_pipeline_time = time.time()

    # Set device (CPU, as per requirement)
    device = torch.device("cpu")
    # Set number of threads for PyTorch CPU operations. os.cpu_count() can be a good default.
    # Ensure it's at least 1.
    num_threads = os.cpu_count() or 1
    torch.set_num_threads(num_threads)
    print(f"Using device: {device} with {torch.get_num_threads()} threads for PyTorch.")

    # 1. Load and Preprocess Data using ORM
    X_tensor, y_tensor, scaler, label_encoder, num_classes = load_and_preprocess_data_orm(
        target_column=COLUMN_TARGET, split_filter='Train' # Assuming 'Train' is the correct SplitType
    )

    if X_tensor is None or num_classes == 0:
        print("[ERROR] Failed to load or preprocess data. Exiting training.")
        return

    # Split data into training and validation sets
    # Stratify by y_tensor to ensure balanced class representation in splits
    stratify_condition = None
    if num_classes > 1 and y_tensor.numel() > 0:
        unique_labels, counts = torch.unique(y_tensor, return_counts=True)
        if len(unique_labels) > 1 and all(c >= 2 for c in counts): # Each class needs at least 2 samples for stratified split
            stratify_condition = y_tensor
        else:
            print("[WARN] Not enough samples per class for stratified split, or only one unique class. Falling back to non-stratified split.")
    else:
        print("[WARN] Not enough classes or no data for stratified split. Falling back to non-stratified split.")

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=stratify_condition
        )
    except ValueError as e:
        print(f"[WARN] Stratified split failed unexpectedly ({e}), falling back to non-stratified split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )

    # Save the fitted scaler and label_encoder
    try:
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"Scaler saved to {SCALER_SAVE_PATH}")
        joblib.dump(label_encoder, LABEL_ENCODER_SAVE_PATH)
        print(f"LabelEncoder saved to {LABEL_ENCODER_SAVE_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save scaler/label_encoder: {e}")


    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model, Optimizer, Criterion, Scheduler
    current_input_dim = X_train.shape[1]
    model = TabularTransformer(
        input_dim=current_input_dim, d_model=D_MODEL, n_head=N_HEAD,
        num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT
    ).to(device)

    # Attempt to compile the model with torch.compile for potential speedup (PyTorch 2.0+)
    # mode="reduce-overhead" is good for CPU and small models
    if hasattr(torch, 'compile'):
        print("Attempting to compile model with torch.compile...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled successfully.")
        except Exception as e:
            print(f"torch.compile failed: {e}. Proceeding with uncompiled model.")
    else:
        print("torch.compile not available (requires PyTorch 2.0+).")


    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/100)

    # 3. Training Loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_accuracy, _, _, _ = evaluate_model(model, val_dataloader, criterion, device, label_encoder)
        scheduler.step() # Step the scheduler

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1:03}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Time: {epoch_duration:.2f}s")

        if pipe_to_ui:
            try:
                pipe_to_ui.send({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy})
            except Exception as e:
                print(f"[WARN] Failed to send data to UI pipe: {e}")
                # pipe_to_ui = None # Stop trying if it fails

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"New best model saved to {BEST_MODEL_SAVE_PATH} (Val Acc: {best_val_accuracy:.4f})")

    # 4. Save Final Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nFinal trained model saved to {MODEL_SAVE_PATH}")

    # 5. Final Evaluation and Saving Predictions (using the best model)
    print("\nLoading best model for final evaluation...")
    # Need to re-initialize model structure before loading state_dict if model was compiled
    # Or, save the original (non-compiled) model's state_dict.
    # For simplicity, we load into a fresh instance.
    final_model = TabularTransformer(
        input_dim=current_input_dim, d_model=D_MODEL, n_head=N_HEAD,
        num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT
    ).to(device)
    final_model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=device))
    # If compiled model was saved, this might need adjustment or saving the original _orig_mod.
    # If torch.compile was used, the saved state_dict is for the *original* model.
    # The loaded final_model is the FP32 version.

    # Note: torch.compile and quantization can sometimes interact in complex ways.
    # Typically, you quantize the original, uncompiled model.
    # If final_model was compiled, we might need its ._orig_mod for quantization,
    # or simply use the state_dict loaded into a fresh, uncompiled model instance.
    # The quantize_and_save_model function already handles creating a fresh instance.

    compiled_final_model_for_eval = None
    if hasattr(torch, 'compile'):
        print("Re-compiling the best FP32 model for evaluation...")
        try:
            # Use a separate variable for the compiled version for FP32 evaluation
            compiled_final_model_for_eval = torch.compile(final_model, mode="reduce-overhead")
            print("Best FP32 model re-compiled successfully for evaluation.")
        except Exception as e:
            print(f"torch.compile failed for best FP32 model: {e}. Evaluating uncompiled FP32 model.")
            compiled_final_model_for_eval = final_model # Fallback to uncompiled
    else:
        compiled_final_model_for_eval = final_model # No compilation available

    eval_model_fp32 = compiled_final_model_for_eval if compiled_final_model_for_eval else final_model

    print("\nFinal Evaluation on Validation Set (Best FP32 Model):")
    _, final_val_accuracy, val_preds_list, val_targets_list, val_confidences_list = evaluate_model(
        final_model, val_dataloader, criterion, device, label_encoder
    )
    print(f"Best Model Validation Accuracy: {final_val_accuracy*100:.2f}%")
    if val_preds_list and val_targets_list:
        # Decode predictions and targets for classification report
        decoded_preds = label_encoder.inverse_transform(np.array(val_preds_list))
        decoded_targets = label_encoder.inverse_transform(np.array(val_targets_list))
        print(classification_report(decoded_targets, decoded_preds, zero_division=0))

    # 6. Quantize the best FP32 model
    # We use the 'final_model' (which has the loaded best state_dict) for quantization.
    # The calibration_dataloader can be train_dataloader or val_dataloader.
    # Using val_dataloader is common if it's representative.
    quantized_model = quantize_and_save_model(
        original_model=final_model, # This is the nn.Module with best FP32 weights
        calibration_dataloader=val_dataloader, # Or train_dataloader
        device=torch.device("cpu"), # Quantization is for CPU
        save_path=QUANTIZED_MODEL_SAVE_PATH,
        input_dim=current_input_dim,
        d_model=D_MODEL,
        n_head=N_HEAD,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout_rate=DROPOUT # Pass other necessary model args
    )

    # 7. Evaluate Quantized Model (optional, but recommended)
    if quantized_model:
        print("\nEvaluating Quantized Model on Validation Set:")
        # Note: The quantized model is already on CPU.
        # We need to ensure the dataloader also provides CPU tensors for this evaluation.
        # DataLoader should be fine as val_dataset tensors are created on CPU.
        q_val_loss, q_val_accuracy, _, _, _ = evaluate_model(
            quantized_model, val_dataloader, criterion, torch.device("cpu"), label_encoder
        )
        print(f"Quantized Model Validation Loss: {q_val_loss:.4f} | Quantized Model Val Acc: {q_val_accuracy:.4f}")
        # Compare q_val_accuracy with final_val_accuracy (FP32)
        print(f"FP32 Val Acc: {final_val_accuracy:.4f} vs Quantized Val Acc: {q_val_accuracy:.4f}")


    # Save predictions to database (using original unscaled validation features for context)
    # This part remains the same, using predictions from the FP32 model for DB saving.
    # If you wanted to save predictions from the quantized model, you'd re-run evaluation.
    # This requires keeping track of original X_val or re-fetching/re-splitting.
    # For simplicity, let's assume X_val (scaled) is what we have.
    # If you need original values, you'd use scaler.inverse_transform(X_val.cpu().numpy())

    # For saving, we need unscaled X_val features.
    # The `X_val` tensor here is already scaled. We need its original form.
    # We can get it by splitting the original `X` (loaded via ORM as a DataFrame) before scaling.

    # Load the full dataset (unscaled, just features and target) again for this purpose.
    # This is slightly redundant but ensures we have the original values for X_val.
    full_original_df = load_training_data_orm(target_column_name=COLUMN_TARGET, split_type_filter='Train')
    if full_original_df.empty:
        print("[ERROR] Could not load original data for saving predictions. Skipping database save.")
        predictions_to_save = []
    else:
        original_X_df_features = full_original_df[[COLUMN_FEATURE1, COLUMN_FEATURE2]]
        original_y_for_split_df = full_original_df[COLUMN_TARGET]

    # Perform the same train_test_split on original unscaled data to get corresponding X_val_original
    # Ensure y_tensor is available for stratification if used before (it is, as label_encoder is fitted)
    # Need to ensure y_original_for_split is in the same format as y_tensor (i.e., encoded if y_tensor was)
    y_original_for_split = label_encoder.transform(original_y_for_split_df.values.ravel()) if label_encoder else original_y_for_split_df.values.ravel()

    try:
        _, X_val_original_unscaled, _, _ = train_test_split(
            original_X_df_features.values, y_original_for_split, test_size=0.2, random_state=42,
            stratify=y_original_for_split if num_classes > 1 and len(np.unique(y_original_for_split)) > 1 else None
        )
    except ValueError as e:
        print(f"[WARN] Stratified split for original data failed ({e}), falling back to non-stratified.")
        _, X_val_original_unscaled, _, _ = train_test_split(
            original_X_df.values, y_original_for_split, test_size=0.2, random_state=42
        )


    print("\nSaving predictions to database...")
    predictions_to_save = []
    for i in range(len(val_preds_list)):
        # Use original unscaled features for saving if available and correctly aligned
        # This assumes X_val_original_unscaled has the same number of rows as val_preds_list
        if i < len(X_val_original_unscaled):
            feature1_val = X_val_original_unscaled[i, 0]
            feature2_val = X_val_original_unscaled[i, 1]
        else: # Fallback if alignment is off (should not happen with correct splitting)
            feature1_val = 0.0 # Or some placeholder
            feature2_val = 0.0

        predictions_to_save.append((
            feature1_val,
            feature2_val,
            int(val_preds_list[i]), # Saving encoded prediction
            float(val_confidences_list[i])
        ))

    if predictions_to_save: # Ensure there's something to save
        try:
            with get_db_session_context() as db: # Use ORM session
                orm_predictions = [
                    Prediction(
                        Feature1=p[0],
                        Feature2=p[1],
                        Prediction=p[2],
                        Confidence=p[3]
                    ) for p in predictions_to_save
                ]
                db.add_all(orm_predictions)
                db.commit()
            print(f"Successfully saved {len(orm_predictions)} predictions to the database via ORM!")
        except Exception as e: # Catch broader SQLAlchemy errors
            print(f"[DB ERROR] ORM Failed to save predictions: {e}")
            if 'db' in locals() and db: db.rollback()
    else:
        print("No predictions to save.")

    pipeline_duration = time.time() - start_pipeline_time
    print(f"\nTraining pipeline completed in {pipeline_duration:.2f} seconds.")
    print(f"Best validation accuracy achieved: {best_val_accuracy*100:.2f}%")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # This allows the script to be imported without running the training pipeline automatically.
    # The `pipe` argument in `train_model` (now `training_pipeline`) was for UI.
    # If this script is run directly, there's no UI pipe.

    # Optional: Create tables if they don't exist (for development)
    # try:
    #     print("Checking and creating database tables if they don't exist (datamodel)...")
    #     create_tables_if_not_exist() # Now uses centralized config
    # except Exception as e:
    #     print(f"Could not check/create tables from datamodel.py: {e}.")
        # Decide if this is fatal or if the script should attempt to continue.
        # For now, we'll let it continue, assuming tables might exist or be created by text_analysis.py run.

    # Global timer start (if you want to time the whole script execution)
    script_start_time = time.time()
    try:
        training_pipeline(
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            pipe_to_ui=None # No UI pipe when run directly
        )
    except pyodbc.Error: # Catch connection errors at the top level too
        print("\n[FATAL ERROR] Database connection issue prevented training. Please check configuration and server status.")
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred during the pipeline: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    script_elapsed_time = time.time() - script_start_time
    print(f"\nTotal script execution time: {script_elapsed_time:.2f} seconds.")
