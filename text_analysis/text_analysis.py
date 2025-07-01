import pandas as pd
# import pyodbc # No longer directly needed here, SQLAlchemy will handle it
import wikipedia
from duckduckgo_search import DDGS
from sqlalchemy.orm import Session as SQLAlchemySessionType
from sqlalchemy import select
from sentence_transformers import SentenceTransformer, util
import customtkinter as ctk  # Import customtkinter
import tkinter as tk # Keep for basic tk types if needed by ctk or for comparison, though direct use should be replaced
import threading
import time
from googletrans import Translator
import re
import torch # For torch.set_num_threads() and tensor operations
import platform # For OS detection
from functools import lru_cache # For caching
import concurrent.futures # For parallel I/O
import os # For environment variables if used for DB_CONFIG

# Assuming database_models.py is in the parent directory or accessible via PYTHONPATH
# Adjust the import path as necessary based on your project structure.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import DB_CONFIG from database_models, and the models/session functions
from database_models import DB_CONFIG, Document, get_db_session, create_tables_if_not_exist, \
                            save_db_config, test_db_connection, CONFIG_FILE_NAME

# Imports for Tabular Classifier
import numpy as np
import joblib
from big_data_pro.datamodel import TabularTransformer, BEST_MODEL_SAVE_PATH as TABULAR_BEST_FP32_PATH, \
                                QUANTIZED_MODEL_SAVE_PATH as TABULAR_QUANTIZED_PATH, \
                                SCALER_SAVE_PATH as TABULAR_SCALER_PATH, \
                                LABEL_ENCODER_SAVE_PATH as TABULAR_LABEL_ENCODER_PATH, \
                                INPUT_DIM as TABULAR_INPUT_DIM, D_MODEL as TABULAR_D_MODEL, \
                                N_HEAD as TABULAR_N_HEAD, NUM_LAYERS as TABULAR_NUM_LAYERS, DROPOUT as TABULAR_DROPOUT
                                # We need NUM_CLASSES for model instantiation, but that's learned during training.
                                # For inference, we'll load LabelEncoder which knows the classes.


# --- Global Configurations ---
# Determine optimal number of threads, can be adjusted
NUM_CPU_THREADS = torch.get_num_threads() # Use PyTorch's default or set explicitly

# Initialize models and utilities
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
TRANSLATOR = Translator()

# DB_CONFIG is now imported from database_models


# --- Database Utilities (SQLAlchemy based) ---
# (get_text_data_from_db_orm, save_interaction_to_db_orm, get_db_session_context)
# These functions remain as previously defined, using SQLAlchemy and the centralized DB_CONFIG.

@lru_cache(maxsize=1)
def get_text_data_from_db_orm() -> pd.DataFrame:
    """
    Retrieves text data (Id, Description) from the 'Documents' table using SQLAlchemy ORM.
    Caches the result to avoid repeated DB calls for the same initial data.
    """
    try:
        # Ensure SQLAlchemySessionType is used for type hinting if Session is ambiguous
        db: SQLAlchemySessionType = next(get_db_session_context())
        stmt = select(Document.Id, Document.Description).where(Document.Description.isnot(None))
        results = db.execute(stmt).fetchall()
        df = pd.DataFrame(results, columns=['Id', 'Description'])
        return df
    except Exception as e:
        print(f"[ERROR] ORM Data Retrieval: {e}")
        return pd.DataFrame(columns=['Id', 'Description'])
    finally:
        if 'db' in locals() and db:
            db.close()

def save_interaction_to_db_orm(question: str, response: str, source: str, pattern: str):
    """
    Saves the user interaction to the 'Documents' table using SQLAlchemy ORM.
    """
    try:
        db: SQLAlchemySessionType = next(get_db_session_context())
        new_document = Document(
            UserQuestion=question,
            Description=response,
            Source=source,
            SearchPattern=pattern
        )
        db.add(new_document)
        db.commit()
    except Exception as e:
        print(f"[DB ERROR] ORM Saving Interaction: {e}")
        if 'db' in locals() and db:
            db.rollback()
    finally:
        if 'db' in locals() and db:
            db.close()

from contextlib import contextmanager
@contextmanager
def get_db_session_context():
    """Provides a SQLAlchemy session that is automatically closed."""
    db = None
    try:
        db = get_db_session()
        yield db
    finally:
        if db:
            db.close()

# --- External Search Utilities ---
@lru_cache(maxsize=128)
def search_wikipedia(query: str):
    """
    Searches Wikipedia for a given query and returns the summary.
    Caches results to speed up repeated queries.
    """
    try:
        wikipedia.set_lang("en") # Or make language configurable
        return wikipedia.summary(query)
    except wikipedia.exceptions.PageError:
        print(f"[WIKIPEDIA] Page not found for query: {query}")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"[WIKIPEDIA] Disambiguation error for query: {query}. Options: {e.options[:3]}")
        # Optionally, try to fetch the first option or return None
        try:
            return wikipedia.summary(e.options[0])
        except:
            return None
    except Exception as e:
        print(f"[WIKIPEDIA ERROR] Query: {query}, Error: {e}")
        return None

@lru_cache(maxsize=128) # Cache recent related Wikipedia searches
def expand_wikipedia_related(query: str, max_related: int = 5) -> list[str]:
    """
    Finds Wikipedia articles related to the query and returns their summaries.
    Caches results.
    """
    try:
        titles = wikipedia.search(query, results=max_related)
        texts = []
        for title in titles:
            try:
                summary = wikipedia.summary(title)
                if summary: # Ensure summary is not None
                    texts.append(summary)
            except Exception: # Broad catch for individual article summary failures
                continue # Skip this article
        return texts
    except Exception as e:
        print(f"[WIKIPEDIA RELATED ERROR] Query: {query}, Error: {e}")
        return []

@lru_cache(maxsize=128) # Cache recent DuckDuckGo searches
def search_duckduckgo(query: str, max_results: int = 3) -> list[str]:
    """
    Searches DuckDuckGo for a given query and returns a list of text snippets.
    Caches results.
    """
    try:
        with DDGS() as ddgs:
            # Using 'max_results' in ddgs.text might not be directly supported by all versions/backends
            # We will fetch a bit more and slice if needed, or rely on its default behavior
            results = ddgs.text(query, max_results=max_results) # Pass max_results here
            return [r["body"] for r in results if "body" in r][:max_results] # Ensure we limit
    except Exception as e:
        print(f"[DUCKDUCKGO ERROR] Query: {query}, Error: {e}")
        return []

# --- Text Processing and NLP Logic ---
COMMAND_KEYWORDS = set([
    "generate", "create", "summarize", "translate", "calculate", "extract", "analyze",
    "find", "define", "describe", "convert", "build", "compare", "evaluate", "plan", "predict"
])

def _is_valid_instruction_word(word: str) -> bool:
    """Checks if a word is a valid potential instruction (alphabetic, min length)."""
    return bool(re.match("^[a-zA-Z]{4,}$", word))

def is_instructional_query(query: str) -> bool:
    """
    Determines if a query is instructional by checking for keywords.
    Learns new instructional keywords from the first word of translated queries.
    """
    global COMMAND_KEYWORDS # Modifying global set
    lower_query = query.lower()
    if any(word in lower_query for word in COMMAND_KEYWORDS):
        return True

    try:
        translated_text = TRANSLATOR.translate(query, dest='en').text.lower()
        if any(word in translated_text for word in COMMAND_KEYWORDS):
            return True

        first_word_translated = translated_text.split()[0]
        if _is_valid_instruction_word(first_word_translated) and first_word_translated not in COMMAND_KEYWORDS:
            COMMAND_KEYWORDS.add(first_word_translated)
            print(f"[LEARNED] New instruction keyword added: {first_word_translated}")
            # Persist learned keywords? For now, it's session-based.
    except Exception as e:
        print(f"[TRANSLATION ERROR] for instructional query check: {e}")

    return False

def combine_text_fragments(fragments: list[str]) -> str:
    """
    Combines a list of text fragments into a single string, avoiding duplicates.
    """
    seen = set()
    combined_parts = []
    for frag in fragments:
        if frag and frag not in seen: # Ensure fragment is not None or empty
            combined_parts.append(frag)
            seen.add(frag)
    return "\n".join(combined_parts).strip()

def _encode_texts(texts: list[str], user_question_embedding):
    """Helper to encode texts and calculate similarity scores."""
    if not texts:
        return []
    embeddings = SENTENCE_MODEL.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(user_question_embedding, embeddings)[0]
    return [(texts[i], float(scores[i])) for i in range(len(texts))]


# --- Core Response Generation Logic ---
def get_best_expanded_response(user_question: str, df_database_texts: pd.DataFrame):
    """
    Generates the best possible response to a user's question by:
    1. Encoding the user question.
    2. Searching in the existing database texts.
    3. Augmenting with results from Wikipedia and DuckDuckGo (parallelized).
    4. Calculating similarity scores for all gathered fragments.
    5. Combining the most relevant fragments.
    6. Saving the interaction to the database.
    """
    user_question_embedding = SENTENCE_MODEL.encode(user_question, convert_to_tensor=True)

    # 1. Search in Database
    db_texts = df_database_texts["Description"].tolist() if not df_database_texts.empty else []
    # Encode DB texts only if not empty, handle empty case
    db_embeddings = SENTENCE_MODEL.encode(db_texts, convert_to_tensor=True) if db_texts else torch.empty(0)

    if db_embeddings.nelement() > 0: # Check if tensor is not empty
        db_scores = util.cos_sim(user_question_embedding, db_embeddings)[0]
        top_db_results = sorted(
            [(db_texts[i], float(db_scores[i])) for i in range(len(db_texts))],
            key=lambda x: x[1],
            reverse=True
        )
    else:
        top_db_results = []


    # 2. Augment with External Searches (Parallelized I/O)
    external_fragments = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_wiki_base = executor.submit(search_wikipedia, user_question)
        future_wiki_related = executor.submit(expand_wikipedia_related, user_question)
        future_duck_results = executor.submit(search_duckduckgo, user_question)

        wiki_base_summary = future_wiki_base.result()
        if wiki_base_summary:
            external_fragments.append(wiki_base_summary)

        external_fragments.extend(future_wiki_related.result())
        external_fragments.extend(future_duck_results.result())

    # 3. Score External Fragments
    # Filter out None or empty strings before encoding
    valid_external_fragments = [frag for frag in external_fragments if frag and frag.strip()]
    enriched_external_results = _encode_texts(valid_external_fragments, user_question_embedding)

    # 4. Combine and Rank All Candidates
    all_candidates = top_db_results + enriched_external_results
    all_candidates.sort(key=lambda x: x[1], reverse=True) # Sort all by score

    # 5. Determine Final Response
    # Increased confidence threshold, e.g. 0.6 for SentenceTransformer "all-MiniLM-L6-v2"
    # This threshold might need tuning.
    CONFIDENCE_THRESHOLD = 0.6

    # Take top N candidates above threshold, or best overall if none meet threshold
    confident_candidates = [cand for cand in all_candidates if cand[1] >= CONFIDENCE_THRESHOLD]

    source_label = "NoConfidence-Fallback"
    final_response_text = "I'm sorry, I couldn't find a confident answer for that."
    highest_score = 0.0

    if confident_candidates:
        # Select top, e.g., 3, confident candidates to form the response
        best_fragments = [text for text, score in confident_candidates[:3]]
        final_response_text = combine_text_fragments(best_fragments)
        highest_score = confident_candidates[0][1]
        source_label = "ExpandedMix-Confident"
    elif all_candidates: # Fallback to the best available if no one is "confident"
        best_single_candidate_text, best_single_candidate_score = all_candidates[0]
        final_response_text = best_single_candidate_text
        highest_score = best_single_candidate_score
        source_label = f"LowConfidence-BestEffort (Score: {highest_score:.2f})"
        # If even the best is very low, provide a generic message
        if highest_score < 0.2: # Very low confidence threshold
             final_response_text = "I found some information, but I'm not very confident about its relevance. Here it is: " + best_single_candidate_text

    if not final_response_text.strip(): # Ensure response is not empty
        final_response_text = "I could not retrieve relevant information for your query."
        source_label = "NoInformationFound"
        highest_score = 0.0


    if is_instructional_query(user_question) and source_label not in ["NoConfidence-Fallback", "NoInformationFound"]:
        final_response_text = f"As per your instruction, here is the information I found:\n{final_response_text}"

    # 6. Save Interaction
    save_interaction_to_db_orm(user_question, final_response_text, source_label, user_question)

    return final_response_text, highest_score, source_label

# --- GUI Application (Using CustomTkinter) ---
class ChatApp(ctk.CTk):
    """
    A CustomTkinter GUI for the Chatbot application.
    """
    def __init__(self, initial_df: pd.DataFrame): # initial_df comes from get_text_data_from_db_orm
        super().__init__()

        self.title(f"ChatAPT ({platform.system()} - CustomTkinter)")
        self.geometry("800x700") # Increased size for tabs
        # ctk.set_appearance_mode("dark")
        # ctk.set_default_color_theme("blue")

        self.current_df = initial_df.copy() # For chatbot's DB text search

        # --- Main Tab View ---
        self.tab_view = ctk.CTkTabview(self, corner_radius=8)
        self.tab_view.pack(expand=True, fill="both", padx=10, pady=10)

        self.tab_chatbot = self.tab_view.add("Chatbot")
        self.tab_classifier = self.tab_view.add("Tabular Classifier")

        self._create_chatbot_tab(self.tab_chatbot)
        self._create_classifier_tab(self.tab_classifier)

        # Load tabular model components
        self._load_tabular_classifier_assets()

    def _load_tabular_classifier_assets(self):
        """Loads the trained TabularTransformer model, scaler, and label encoder."""
        self.tabular_model = None
        self.tabular_scaler = None
        self.tabular_label_encoder = None
        self.tabular_num_classes = None # Will get from label_encoder

        try:
            # Load Label Encoder first to get num_classes
            if os.path.exists(TABULAR_LABEL_ENCODER_PATH):
                self.tabular_label_encoder = joblib.load(TABULAR_LABEL_ENCODER_PATH)
                self.tabular_num_classes = len(self.tabular_label_encoder.classes_)
                print(f"LabelEncoder loaded. Classes: {self.tabular_label_encoder.classes_}")
            else:
                print(f"[ERROR] LabelEncoder not found at {TABULAR_LABEL_ENCODER_PATH}")
                self.classifier_result_text.set("Error: LabelEncoder not found.")
                return

            # Load Scaler
            if os.path.exists(TABULAR_SCALER_PATH):
                self.tabular_scaler = joblib.load(TABULAR_SCALER_PATH)
                print("Scaler loaded.")
            else:
                print(f"[ERROR] Scaler not found at {TABULAR_SCALER_PATH}")
                self.classifier_result_text.set("Error: Scaler not found.")
                return

            # Load Model (Quantized preferred)
            model_path_to_load = None
            if os.path.exists(TABULAR_QUANTIZED_PATH):
                model_path_to_load = TABULAR_QUANTIZED_PATH
                print(f"Loading Quantized Tabular Model from {model_path_to_load}...")
                # For JIT scripted quantized model
                self.tabular_model = torch.jit.load(model_path_to_load, map_location=torch.device('cpu'))
            elif os.path.exists(TABULAR_QUANTIZED_PATH + ".state_dict"): # Fallback for state_dict
                 model_path_to_load = TABULAR_QUANTIZED_PATH + ".state_dict"
                 print(f"Loading Quantized Tabular Model (state_dict) from {model_path_to_load}...")
                 # Need to instantiate the model structure for quantized state_dict
                 # This requires knowing the exact structure, including qconfig.
                 # For simplicity, we'll prioritize the JIT model or FP32 for now if JIT fails.
                 # TODO: Add robust loading for quantized state_dict if needed.
                 # For now, if JIT fails, try FP32.
                 print("[WARN] Loading quantized state_dict is complex, attempting FP32 model instead for now if JIT scripted version not found.")
                 model_path_to_load = None # Reset to try FP32

            if not self.tabular_model and os.path.exists(TABULAR_BEST_FP32_PATH):
                model_path_to_load = TABULAR_BEST_FP32_PATH
                print(f"Loading FP32 Tabular Model from {model_path_to_load}...")
                # Instantiate FP32 model structure
                self.tabular_model = TabularTransformer(
                    input_dim=TABULAR_INPUT_DIM, d_model=TABULAR_D_MODEL, n_head=TABULAR_N_HEAD,
                    num_layers=TABULAR_NUM_LAYERS, num_classes=self.tabular_num_classes, dropout=TABULAR_DROPOUT
                )
                self.tabular_model.load_state_dict(torch.load(model_path_to_load, map_location=torch.device('cpu')))

            if self.tabular_model:
                self.tabular_model.eval() # Set to evaluation mode
                print("Tabular model loaded successfully.")
                if hasattr(self, 'classifier_status_label'): # Update status if UI element exists
                    self.classifier_status_label.configure(text="Model loaded.", text_color="green")
            else:
                print(f"[ERROR] No Tabular Model found at specified paths.")
                if hasattr(self, 'classifier_status_label'):
                     self.classifier_status_label.configure(text="Error: Model not found.", text_color="red")


        except Exception as e:
            print(f"[ERROR] Loading tabular classifier assets: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'classifier_status_label'):
                self.classifier_status_label.configure(text=f"Error loading model: {e}", text_color="red")
            self.tabular_model = None # Ensure it's None on error

    def _create_chatbot_tab(self, tab_master):
        """Creates the UI elements for the chatbot tab."""
        tab_master.grid_columnconfigure(0, weight=1)
        tab_master.grid_rowconfigure(0, weight=1)
        tab_master.grid_rowconfigure(1, weight=0)

        self.text_area = ctk.CTkTextbox(
            master=tab_master, wrap="word", state='disabled',
            font=("Segoe UI", 13), corner_radius=6, border_width=1
        )
        self.text_area.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        input_frame_chat = ctk.CTkFrame(master=tab_master, fg_color="transparent")
        input_frame_chat.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        input_frame_chat.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            master=input_frame_chat, placeholder_text="Ask me anything...",
            font=("Segoe UI", 13), corner_radius=6, border_width=1
        )
        self.entry.grid(row=0, column=0, padx=(0,10), sticky="ew")
        self.entry.bind("<Return>", self._on_send_pressed)

        self.send_btn = ctk.CTkButton(
            master=input_frame_chat, text="➤", command=self._on_send_pressed,
            width=50, font=("Segoe UI", 16, "bold"), corner_radius=6
        )
        self.send_btn.grid(row=0, column=1, sticky="e")

    def _create_classifier_tab(self, tab_master):
        """Creates the UI elements for the tabular classifier tab."""
        tab_master.grid_columnconfigure(0, weight=1) # Allow content to center/expand if needed
        # Configure rows for labels, entries, button, result
        tab_master.grid_rowconfigure(0, weight=0, pad=10) # Feature1 Label
        tab_master.grid_rowconfigure(1, weight=0, pad=5)  # Feature1 Entry
        tab_master.grid_rowconfigure(2, weight=0, pad=10) # Feature2 Label
        tab_master.grid_rowconfigure(3, weight=0, pad=5)  # Feature2 Entry
        tab_master.grid_rowconfigure(4, weight=0, pad=20) # Predict Button
        tab_master.grid_rowconfigure(5, weight=0, pad=10) # Result Label
        tab_master.grid_rowconfigure(6, weight=0, pad=10) # Status Label
        tab_master.grid_rowconfigure(7, weight=1)         # Spacer to push content up

        ctk.CTkLabel(tab_master, text="Feature 1:", font=("Segoe UI", 12)).grid(row=0, column=0, sticky="w", padx=20)
        self.feature1_entry = ctk.CTkEntry(tab_master, placeholder_text="Enter value for Feature 1 (e.g., 0.5)", font=("Segoe UI", 13), corner_radius=6, border_width=1)
        self.feature1_entry.grid(row=1, column=0, padx=20, sticky="ew")

        ctk.CTkLabel(tab_master, text="Feature 2:", font=("Segoe UI", 12)).grid(row=2, column=0, sticky="w", padx=20)
        self.feature2_entry = ctk.CTkEntry(tab_master, placeholder_text="Enter value for Feature 2 (e.g., -1.2)", font=("Segoe UI", 13), corner_radius=6, border_width=1)
        self.feature2_entry.grid(row=3, column=0, padx=20, sticky="ew")

        self.predict_btn = ctk.CTkButton(tab_master, text="Predict Class", command=self._on_predict_button_pressed, font=("Segoe UI", 14, "bold"), corner_radius=6)
        self.predict_btn.grid(row=4, column=0, padx=20, pady=20)

        self.classifier_result_text = ctk.StringVar(value="Prediction will appear here.")
        ctk.CTkLabel(tab_master, textvariable=self.classifier_result_text, font=("Segoe UI", 14, "bold")).grid(row=5, column=0, padx=20, pady=10)

        self.classifier_status_label = ctk.CTkLabel(tab_master, text="Loading model...", font=("Segoe UI", 10))
        self.classifier_status_label.grid(row=6, column=0, padx=20, pady=5)


    def _on_predict_button_pressed(self):
        """Handles the predict button press for the tabular classifier."""
        if not self.tabular_model or not self.tabular_scaler or not self.tabular_label_encoder:
            self.classifier_result_text.set("Error: Model or preprocessors not loaded.")
            print("[ERROR] Prediction attempt failed: model/scaler/encoder not loaded.")
            return

        try:
            f1_str = self.feature1_entry.get()
            f2_str = self.feature2_entry.get()

            if not f1_str or not f2_str:
                self.classifier_result_text.set("Please enter values for both features.")
                return

            feature1 = float(f1_str)
            feature2 = float(f2_str)
        except ValueError:
            self.classifier_result_text.set("Error: Features must be valid numbers.")
            return

        try:
            # Preprocess input
            input_data = np.array([[feature1, feature2]])
            scaled_input_data = self.tabular_scaler.transform(input_data)
            input_tensor = torch.tensor(scaled_input_data, dtype=torch.float32).to(torch.device('cpu'))

            # Predict
            with torch.no_grad():
                output = self.tabular_model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

            predicted_class_label = self.tabular_label_encoder.inverse_transform(predicted_idx.cpu().numpy())[0]

            result_str = f"Predicted Class: {predicted_class_label}\nConfidence: {confidence.item()*100:.2f}%"
            self.classifier_result_text.set(result_str)
            print(f"Prediction: {result_str} for input [{feature1}, {feature2}]")

        except Exception as e:
            self.classifier_result_text.set(f"Error during prediction: {e}")
            print(f"[ERROR] Tabular prediction error: {e}")
            import traceback
            traceback.print_exc()


    def _append_message_to_chat(self, sender: str, message: str):
        """Appends a message to the chat text area (used by chatbot tab)."""
        self.text_area.configure(state='normal')
        self.text_area.insert("end", f"{sender}: {message}\n\n")
        self.text_area.see("end")
        self.text_area.configure(state='disabled')

    def _show_thinking_indicator(self, thinking: bool):
        """Shows or hides a 'thinking...' indicator."""
        if thinking:
            self._append_message_to_chat("🤖 AI", "Thinking...")
        # Else: The actual answer will follow, effectively removing/replacing the indicator.

    def _process_question_thread(self, question: str):
        """
        Handles question processing in a separate thread to keep UI responsive.
        """
        self.after(0, self._show_thinking_indicator, True)

        response_text, score, source = get_best_expanded_response(question, self.current_df)

        self.after(0, self._show_thinking_indicator, False) # Call to potentially clear or be overwritten
        self.after(0, self._append_message_to_chat, "🤖 AI", f"{response_text}\n(Source: {source}, Score: {score:.2f})")

        if source not in ["NoConfidence-Fallback", "NoInformationFound", "NoInformationFound"]:
            new_row_data = [{"Id": -1, "Description": response_text}]
            new_row_df = pd.DataFrame(new_row_data)
            self.current_df = pd.concat([self.current_df, new_row_df], ignore_index=True)

    def _on_send_pressed(self, event=None):
        """Handles the send button press or Enter key in the entry field."""
        question = self.entry.get().strip()
        if not question:
            return

        self.entry.delete(0, "end") # Use "end" for CTkEntry
        self._append_message_to_chat("👤 You", question)

        thread = threading.Thread(target=self._process_question_thread, args=(question,))
        thread.daemon = True
        thread.start()

# --- Main Application Launch ---
if __name__ == "__main__":
    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

    print(f"Application running on {platform.system()} with {NUM_CPU_THREADS} PyTorch CPU threads.")

    try:
        print("Checking and creating database tables if they don't exist (text_analysis)...")
        create_tables_if_not_exist() # Now uses centralized config, no args needed
        print("Table check/creation process complete (text_analysis).")
    except Exception as e:
        print(f"Could not check/create tables from text_analysis.py: {e}. Ensure DB server is running and accessible.")
        print("SQLAlchemy and DB drivers must be correctly installed, and DB accessible.")
        sys.exit(1)

    initial_db_data = get_text_data_from_db_orm() # Use the ORM based function
    if initial_db_data.empty:
        print("[WARN] No initial data found in the database via ORM. Chatbot will rely on external sources.")
        initial_db_data = pd.DataFrame(columns=["Id", "Description"])

    print("[INFO] Model ready. Launching GUI application with CustomTkinter...")
    app_gui = ChatApp(initial_db_data) # Pass initial_df to the constructor
    app_gui.mainloop()
