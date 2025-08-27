# src/rag_pipeline.py

import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration (No changes here) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
FINE_TUNED_MODEL_PATH = PROJECT_ROOT / "final_adapter_model"
DOCUMENTS_PATH = PROJECT_ROOT / "datasets" / "IN-Ext" / "judgement"

# --- Helper Functions (No changes here) ---
def split_text_into_chunks(text: str, tokenizer: AutoTokenizer, chunk_size: int = 1500, chunk_overlap: int = 200):
    tokens = tokenizer.encode(text)
    token_chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        token_chunks.append(tokens[i:i + chunk_size])
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]

class SimpleRAG:
    # ... (This class remains exactly the same) ...
    def __init__(self, documents_path):
        self.documents_path = documents_path
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self._build_index()
    def _build_index(self):
        doc_files = list(self.documents_path.glob("*.txt"))
        if not doc_files: return
        for doc_path in doc_files:
            self.documents.append(doc_path.read_text(encoding="utf-8", errors="ignore"))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    def retrieve(self, query: str, top_k: int = 1):
        if self.tfidf_matrix is None: return None
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        if similarities[top_k_indices[0]] == 0: return None
        return self.documents[top_k_indices[0]]

def load_model(base_model_name, fine_tuned_path):
    # ... (This function remains exactly the same) ...
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if not fine_tuned_path.exists():
        logging.error(f"FATAL: Model path not found at {fine_tuned_path}")
        return None, None
    model = PeftModel.from_pretrained(model, str(fine_tuned_path))
    return model, tokenizer

def generate_response(model, tokenizer, system_message: str, user_prompt: str):
    # ... (This function remains exactly the same) ...
    full_prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=2048, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2, do_sample=True, temperature=0.4, top_p=0.9)
    response_ids = outputs[0][len(inputs['input_ids'][0]):]
    return tokenizer.decode(response_ids, skip_special_tokens=True)

# --- AGENT DEFINITIONS ---

def summarizer_agent(query: str, rag_pipeline: SimpleRAG, model, tokenizer):
    """Agent 1: Handles the full Map-Reduce summarization task."""
    logging.info(f"Summarizer Agent activated for query: '{query}'")
    retrieved_document = rag_pipeline.retrieve(query)
    if not retrieved_document:
        return "Could not retrieve a relevant document for that query. Please try different keywords.", None

    # MAP STEP
    logging.info("Document is long. Splitting into chunks...")
    chunks = split_text_into_chunks(retrieved_document, tokenizer)
    chunk_summaries = []
    map_system_prompt = "You are an expert legal assistant. Your task is to provide a clear and concise summary of the provided legal document excerpt."
    for i, chunk in enumerate(chunks):
        logging.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
        chunk_summary = generate_response(model, tokenizer, system_message=map_system_prompt, user_prompt=chunk)
        chunk_summaries.append(chunk_summary)

    # REDUCE STEP
    logging.info("Combining chunk summaries into a final summary...")
    combined_summary_text = "\n\n".join(chunk_summaries)
    reduce_system_prompt = "You are an expert editor. Synthesize the following summaries of a legal case into a single, final, coherent summary."
    final_summary = generate_response(model, tokenizer, system_message=reduce_system_prompt, user_prompt=combined_summary_text)
    
    return f"--- FINAL CONSOLIDATED SUMMARY ---\n{final_summary}\n----------------------------------\n", final_summary

def qa_agent(question: str, context_summary: str, model, tokenizer):
    """Agent 2: Answers questions based ONLY on the provided summary context."""
    logging.info(f"Q&A Agent activated for question: '{question}'")
    
    system_prompt = "You are a helpful chatbot. Answer the user's question based *only* on the provided summary. If the answer is not in the summary, say 'I cannot answer that based on the provided summary.'"
    user_prompt = f"Based on this summary:\n---\n{context_summary}\n---\n\nAnswer this question: {question}"
    
    answer = generate_response(model, tokenizer, system_message=system_prompt, user_prompt=user_prompt)
    return f"--- Answer ---\n{answer}\n--------------\n"

# --- MAIN ORCHESTRATOR LOOP ---
if __name__ == "__main__":
    model, tokenizer = load_model(BASE_MODEL_NAME, FINE_TUNED_MODEL_PATH)
    
    if model and tokenizer:
        rag_pipeline = SimpleRAG(documents_path=DOCUMENTS_PATH)
        
        if rag_pipeline.tfidf_matrix is not None:
            # State management: keeps track of the current summary
            current_summary_context = None 
            
            while True:
                if current_summary_context is None:
                    # STATE 1: No summary exists. We are in "Summarization Mode".
                    user_input = input("\nEnter a query to find and summarize a legal document (or type 'exit'): ")
                    if user_input.lower() == 'exit': break
                    
                    # Call the Summarizer Agent
                    response, summary_for_context = summarizer_agent(user_input, rag_pipeline, model, tokenizer)
                    print(response)
                    current_summary_context = summary_for_context # Update state
                
                else:
                    # STATE 2: A summary exists. We are in "Q&A Mode".
                    user_input = input("\nAsk a follow-up question about the summary, type 'new' for a new document, or 'exit': ")
                    if user_input.lower() == 'exit': break
                    if user_input.lower() == 'new':
                        current_summary_context = None # Reset state to go back to Summarization Mode
                        continue

                    # Call the Q&A Agent
                    response = qa_agent(user_input, current_summary_context, model, tokenizer)
                    print(response)