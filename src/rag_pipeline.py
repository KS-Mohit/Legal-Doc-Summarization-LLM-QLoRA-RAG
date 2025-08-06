# src/rag_pipeline.py

import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
FINE_TUNED_MODEL_PATH = PROJECT_ROOT / "final_adapter_model"
DOCUMENTS_PATH = PROJECT_ROOT / "datasets" / "IN-Ext" / "judgement"
# ---

# --- MAP-REDUCE HELPER FUNCTION ---
def split_text_into_chunks(text: str, tokenizer: AutoTokenizer, chunk_size: int = 1500, chunk_overlap: int = 200):
    """
    Splits a long text into smaller chunks based on token count, with overlap.
    This is the first part of the "Map" step.
    """
    tokens = tokenizer.encode(text)
    token_chunks = []
    # Use a sliding window to create chunks
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk = tokens[i:i + chunk_size]
        token_chunks.append(chunk)
    
    # Decode token chunks back to text strings
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]
    return text_chunks

# --- RAG Components (No changes needed here) ---
class SimpleRAG:
    # ... (This class remains exactly the same as before) ...
    """A simple Retrieval-Augmented Generation pipeline."""
    def __init__(self, documents_path):
        self.documents_path = documents_path
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        logging.info("Initializing RAG pipeline...")
        self._build_index()

    def _build_index(self):
        """Loads original documents and builds the TF-IDF vector index."""
        logging.info(f"Loading original documents from: {self.documents_path}")
        doc_files = list(self.documents_path.glob("*.txt"))
        if not doc_files:
            logging.error(f"No documents found in {self.documents_path}. RAG will not work.")
            return
        for doc_path in doc_files:
            try:
                self.documents.append(doc_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                logging.warning(f"Could not read {doc_path.name}: {e}")
        logging.info(f"Building TF-IDF index for {len(self.documents)} documents...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        logging.info("RAG index built successfully.")

    def retrieve(self, query: str, top_k: int = 1):
        """Retrieves the top_k most relevant documents for a query."""
        if self.tfidf_matrix is None: return None
        logging.info(f"Retrieving document for query: '{query}'")
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        if similarities[top_k_indices[0]] == 0:
            logging.warning("No relevant document found for the query.")
            return None
        retrieved_doc = self.documents[top_k_indices[0]]
        logging.info("Document retrieved.")
        return retrieved_doc

# --- LLM Loading and Generation ---
def load_model(base_model_name, fine_tuned_path):
    # ... (This function remains exactly the same as before) ...
    """Loads the base model and applies the fine-tuned LoRA adapters."""
    logging.info(f"Loading base model: {base_model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Loading LoRA adapters from: {fine_tuned_path}")
    if not fine_tuned_path.exists():
        logging.error(f"FATAL: Fine-tuned model path not found at {fine_tuned_path}")
        return None, None
    model = PeftModel.from_pretrained(model, str(fine_tuned_path))
    logging.info("Model and adapters loaded successfully.")
    return model, tokenizer

# --- UPDATED Generation Function ---
def generate_response(model, tokenizer, system_message: str, user_prompt: str):
    """
    A more flexible generation function that takes a system message and user prompt.
    This will be used for both the 'Map' and 'Reduce' steps.
    """
    # This prompt format MUST match the one used during training.
    full_prompt = f"""<|system|>
{system_message}</s>
<|user|>
{user_prompt}</s>
<|assistant|>
"""
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=2048, truncation=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )
    response_ids = outputs[0][len(inputs['input_ids'][0]):]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text

# --- UPDATED Main Execution with Map-Reduce Logic ---
if __name__ == "__main__":
    model, tokenizer = load_model(BASE_MODEL_NAME, FINE_TUNED_MODEL_PATH)
    
    if model and tokenizer:
        rag_pipeline = SimpleRAG(documents_path=DOCUMENTS_PATH)
        
        if rag_pipeline.tfidf_matrix is not None:
            while True:
                query = input("\nEnter a query (e.g., case name or details), or type 'exit' to quit: ")
                if query.lower() == 'exit': break
                
                retrieved_document = rag_pipeline.retrieve(query)
                
                if retrieved_document:
                    # --- MAP-REDUCE LOGIC STARTS HERE ---
                    
                    # 1. MAP STEP
                    logging.info("Document is long. Splitting into chunks for summarization...")
                    chunks = split_text_into_chunks(retrieved_document, tokenizer)
                    chunk_summaries = []
                    
                    map_system_prompt = "You are an expert legal assistant. Your task is to provide a clear and concise summary of the provided legal document excerpt."

                    for i, chunk in enumerate(chunks):
                        logging.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
                        chunk_summary = generate_response(model, tokenizer, system_message=map_system_prompt, user_prompt=chunk)
                        chunk_summaries.append(chunk_summary)
                        print(f"--- Summary of Chunk {i+1} ---\n{chunk_summary}\n--------------------")

                    # 2. REDUCE STEP
                    logging.info("Combining chunk summaries into a final summary...")
                    combined_summary_text = "\n\n".join(chunk_summaries)
                    
                    reduce_system_prompt = "You are an expert editor. You will be given a series of summaries about a single legal case. Your job is to synthesize them into a single, final, coherent summary of the entire case."
                    
                    final_summary = generate_response(model, tokenizer, system_message=reduce_system_prompt, user_prompt=combined_summary_text)

                    print("\n--- FINAL CONSOLIDATED SUMMARY ---")
                    print(final_summary)
                    print("----------------------------------\n")
                else:
                    print("Could not retrieve a relevant document for that query. Please try different keywords.")