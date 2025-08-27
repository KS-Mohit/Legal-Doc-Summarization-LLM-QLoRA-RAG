# src/app.py

import logging
import json
import os
import requests
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)

# --- Global Variables ---
model, tokenizer, rag_pipeline = None, None, None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Using the NON-STREAMING endpoint for Gemini
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

# --- Configuration Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
FINE_TUNED_MODEL_PATH = PROJECT_ROOT / "final_adapter_model"
DOCUMENTS_PATH = PROJECT_ROOT / "datasets" / "IN-Ext" / "judgement"

# --- Helper Functions and Classes (No changes) ---
def split_text_into_chunks(text: str, tokenizer: AutoTokenizer, chunk_size: int = 1500, chunk_overlap: int = 200):
    tokens = tokenizer.encode(text)
    token_chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        token_chunks.append(tokens[i:i + chunk_size])
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]

class SimpleRAG:
    def __init__(self, documents_path):
        self.documents, self.vectorizer, self.tfidf_matrix = [], TfidfVectorizer(stop_words='english'), None
        self._build_index(documents_path)
    def _build_index(self, documents_path):
        doc_files = list(documents_path.glob("*.txt"))
        if not doc_files: return
        for doc_path in doc_files: self.documents.append(doc_path.read_text(encoding="utf-8", errors="ignore"))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    def retrieve(self, query: str):
        if self.tfidf_matrix is None: return None
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_indices = np.argsort(similarities)[-1:][::-1]
        if similarities[top_k_indices[0]] == 0: return None
        return self.documents[top_k_indices[0]]

def load_model_and_rag(base_model_name, fine_tuned_path, docs_path):
    global model, tokenizer, rag_pipeline
    logging.info("Loading local TinyLlama model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, str(fine_tuned_path))
    rag_pipeline = SimpleRAG(docs_path)
    logging.info("Local model and RAG pipeline loaded successfully.")

def generate_local_response(system_message: str, user_prompt: str):
    full_prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=2048, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.3, temperature=0.2, top_p=0.9, do_sample=True, eos_token_id=tokenizer.eos_token_id, early_stopping=True)
    response_ids = outputs[0][len(inputs['input_ids'][0]):]
    return tokenizer.decode(response_ids, skip_special_tokens=True)

# --- API Endpoints ---

@app.route('/query', methods=['GET'])
def handle_query_stream():
    query = request.args.get('query')
    def stream_generator():
        try:
            yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieving relevant document...'})}\n\n"
            retrieved_document = rag_pipeline.retrieve(query)
            if not retrieved_document:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Could not retrieve a relevant document.'})}\n\n"
                return

            chunks = split_text_into_chunks(retrieved_document, tokenizer)
            chunk_summaries = []
            map_system_prompt = "You are an expert legal assistant. Your task is to provide a clear and concise summary of the provided legal document excerpt."
            
            for i, chunk in enumerate(chunks):
                yield f"data: {json.dumps({'type': 'status', 'content': f'Summarizing chunk {i+1}/{len(chunks)}...'})}\n\n"
                chunk_summary = generate_local_response(system_message=map_system_prompt, user_prompt=chunk)
                chunk_summaries.append(chunk_summary)
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_summary, 'chunk_num': i+1, 'total_chunks': len(chunks)})}\n\n"
            
            combined_context = "\n\n".join(chunk_summaries)
            yield f"data: {json.dumps({'type': 'final', 'content': 'Document analysis complete. You can now ask questions.', 'context_for_qa': combined_context})}\n\n"
        except Exception as e:
            logging.error(f"Error during stream generation: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred on the server.'})}\n\n"
    return Response(stream_generator(), mimetype='text/event-stream')

# CORRECTED: This is now a simple, non-streaming POST request
@app.route('/ask', methods=['POST'])
def handle_ask():
    data = request.json
    question = data.get('question')
    context = data.get('context')

    if not question or not context:
        return jsonify({'error': 'Question and context are required'}), 400

    logging.info(f"Q&A Agent activated with Gemini for question: '{question}'")
    
    system_prompt = "You are a helpful legal chatbot. Based ONLY on the provided context below, answer the user's question. If the answer is not found, state that clearly."
    payload = {
        "contents": [{"parts": [{"text": f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {question}"}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.3, "topP": 0.9}
    }
    
    try:
        response = requests.post(GEMINI_API_URL, json=payload)
        response.raise_for_status() 
        result = response.json()
        answer = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Sorry, I could not generate an answer.')
        return jsonify({'answer': answer})
    except Exception as e:
        logging.error(f"Gemini API request failed: {e}")
        return jsonify({'error': 'Failed to communicate with the Gemini API.'}), 500

if __name__ == '__main__':
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    load_model_and_rag(BASE_MODEL_NAME, FINE_TUNED_MODEL_PATH, DOCUMENTS_PATH)
    app.run(host='0.0.0.0', port=5000)
