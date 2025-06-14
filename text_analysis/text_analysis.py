import pandas as pd
import pymssql
import wikipedia
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util
import tkinter as tk
from tkinter import ttk
import threading
import time
from googletrans import Translator
import re

model = SentenceTransformer("all-MiniLM-L6-v2")
translator = Translator()

# Get responses from SQL Server
def get_text_data_from_sql():
    try:
        conn = pymssql.connect(
            server='localhost.localdomain',
            user='sa',
            password='brian04271208@',
            database='TransformerNeuronDB'
        )
        query = "SELECT Id, Description FROM Documents WHERE Description IS NOT NULL"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[ERROR] {e}")
        return pd.DataFrame()

# Save interaction to DB
def save_to_db(question, response, source, pattern):
    try:
        conn = pymssql.connect(
            server='localhost.localdomain',
            user='sa',
            password='brian04271208@',
            database='TransformerNeuronDB'
        )
        cursor = conn.cursor()
        insert_query = "INSERT INTO Documents (UserQuestion, Description, Source, SearchPattern) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_query, (question, response, source, pattern))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] {e}")

# Wikipedia search
def search_wikipedia(query):
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(query)
    except:
        return None

# Related Wikipedia topics
def expand_wikipedia_related(query):
    try:
        titles = wikipedia.search(query, results=10)
        texts = []
        for title in titles:
            try:
                summary = wikipedia.summary(title)
                texts.append(summary)
            except:
                continue
        return texts
    except:
        return []

# DuckDuckGo search
def search_duckduckgo(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [r["body"] for r in results if "body" in r]
    except Exception as e:
        print(f"[DUCKDUCKGO ERROR] {e}")
        return []

# Instructional Parsing - Detect action-based queries
command_keywords = set([
    "generate", "create", "summarize", "translate", "calculate", "extract", "analyze",
    "find", "define", "describe", "convert", "build", "compare", "evaluate", "plan", "predict"
])

def is_valid_instruction(instruction):
    return bool(re.match("^[a-zA-Z]{4,}$", instruction))

def is_instructional_query(query):
    global command_keywords
    lower_query = query.lower()
    for word in command_keywords:
        if word in lower_query:
            return True

    translated = translator.translate(query, dest='en').text.lower()
    for word in command_keywords:
        if word in translated:
            return True

    new_word = translated.split()[0]
    if new_word not in command_keywords and is_valid_instruction(new_word):
        command_keywords.add(new_word)
        print(f"[LEARNED] New instruction keyword added: {new_word}")

    return False

# Combine fragments
def combine_responses(base_fragments):
    combined = ""
    seen = set()
    for frag in base_fragments:
        if frag not in seen:
            combined += frag + "\n"
            seen.add(frag)
    return combined.strip()

# Main logic with introspection + instruction handling
def get_best_expanded_response(user_question, df):
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    db_texts = df["Description"].tolist()
    db_embeddings = model.encode(db_texts, convert_to_tensor=True)
    db_scores = util.cos_sim(user_embedding, db_embeddings)[0]
    top_db = [(db_texts[i], float(db_scores[i])) for i in db_scores.argsort(descending=True)]

    wiki_base = search_wikipedia(user_question)
    wiki_related = expand_wikipedia_related(user_question)
    duck_results = search_duckduckgo(user_question)

    all_fragments = []
    if wiki_base:
        all_fragments.append(wiki_base)
    all_fragments += wiki_related + duck_results

    enriched = []
    for frag in all_fragments:
        frag_embedding = model.encode(frag, convert_to_tensor=True)
        sim = float(util.cos_sim(user_embedding, frag_embedding)[0])
        enriched.append((frag, sim))

    all_candidates = top_db + enriched
    confident = [(text, score) for text, score in all_candidates if score >= 1.0]

    if not confident:
        best_candidate = max(all_candidates, key=lambda x: x[1])
        save_to_db(user_question, best_candidate[0], "LowConfidence-BestEffort", user_question)
        return best_candidate[0], best_candidate[1], "LowConfidence-BestEffort"

    confident.sort(key=lambda x: x[1], reverse=True)
    best_fragments = [frag for frag, _ in confident[:3]]
    final_response = combine_responses(best_fragments)

    if is_instructional_query(user_question):
        final_response = f"As requested, here is the result:\n{final_response}"

    save_to_db(user_question, final_response, "ExpandedMix", user_question)
    return final_response, confident[0][1], "ExpandedMix"

# GUI App
class ChatApp:
    def __init__(self, root, df):
        self.root = root
        self.root.title("ChatAPT")
        self.df = df

        self.root.configure(bg="#121212")

        self.chat_frame = tk.Frame(root, bg="#121212")
        self.chat_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.text_area = tk.Text(self.chat_frame, wrap=tk.WORD, width=70, height=20, state='disabled',
                                 font=("Segoe UI", 11), bg="#1f1f1f", fg="#e4e4e4", insertbackground="white",
                                 borderwidth=0, relief="flat", padx=12, pady=8)
        self.text_area.pack(padx=10, pady=10, fill="both", expand=True)

        self.input_frame = tk.Frame(root, bg="#121212")
        self.input_frame.pack(fill="x", padx=10, pady=(0, 12))

        self.entry = tk.Entry(self.input_frame, font=("Segoe UI", 11), bg="#1f1f1f", fg="white",
                              insertbackground="white", relief="flat")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=8)
        self.entry.bind("<Return>", self.handle_question)

        self.send_btn = tk.Button(self.input_frame, text="➤", command=self.handle_question, bg="#007aff", fg="white",
                                  font=("Segoe UI", 12, "bold"), relief="flat", activebackground="#005bb5",
                                  padx=16, pady=8)
        self.send_btn.pack(side="right")

    def append_text(self, sender, message):
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, f"{sender}: {message}\n\n")
        self.text_area.see(tk.END)
        self.text_area.config(state='disabled')

    def show_thinking(self):
        self.append_text("🤖 AI", "Thinking")
        for _ in range(3):
            time.sleep(0.5)
            self.text_area.config(state='normal')
            self.text_area.insert(tk.END, ".")
            self.text_area.see(tk.END)
            self.text_area.config(state='disabled')
        time.sleep(0.5)
        self.text_area.config(state='normal')
        self.text_area.delete("end-3l", "end")
        self.text_area.config(state='disabled')

    def handle_question(self, event=None):
        question = self.entry.get().strip()
        self.entry.delete(0, tk.END)

        if question:
            self.append_text("You", question)

            def process():
                self.show_thinking()
                response, score, source = get_best_expanded_response(question, self.df)
                self.append_text("🤖 AI", f"{response}\n(Source: {source}, Score: {score:.2f})")

                new_row = {"Description": response}
                self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

            threading.Thread(target=process).start()

# Launch
if __name__ == "__main__":
    df = get_text_data_from_sql()
    if not df.empty:
        print("[INFO] Model ready. Launching app...")
        root = tk.Tk()
        root.geometry("700x600")
        root.resizable(False, False)
        app = ChatApp(root, df)
        root.mainloop()
    else:
        print("[WARN] No data found in the database.")
