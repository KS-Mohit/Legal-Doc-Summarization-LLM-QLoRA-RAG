# Advanced Legal Document Summarizer with TinyLlama, QLoRA, and Map-Reduce RAG

This project presents a robust, end-to-end pipeline for summarizing long and complex legal documents. It leverages a compact yet powerful Large Language Model (LLM), **TinyLlama-1.1B**, fine-tuned using the memory-efficient **QLoRA** technique.

The core innovation of this project is a **Map-Reduce RAG pipeline**, specifically designed to overcome the context-window limitations of smaller models. This allows the system to process and generate comprehensive summaries for legal documents of arbitrary length, a significant improvement over standard RAG implementations that rely on simple truncation.

## Key Features

* **Efficient Fine-Tuning:** Utilizes **QLoRA** to fine-tune the TinyLlama model on consumer-grade GPUs, making advanced NLP accessible.
* **Long Document Handling:** Implements a **Map-Reduce** strategy to process documents that exceed the model's 2048-token context window, ensuring no information is lost.
* **End-to-End Pipeline:** Covers the entire workflow from data preprocessing and model fine-tuning to interactive retrieval and summarization.
* **Optimized for Small Models:** The entire workflow, including the prompt formats and generation parameters, has been tailored to get the best performance from the compact TinyLlama model.
* **Robust Codebase:** All scripts use modern Python practices, including `pathlib` for reliable path management and a clear, modular structure.

## Architecture: The Map-Reduce RAG Pipeline

The system uses an advanced RAG architecture to handle long documents effectively.

```
User Query -> [TF-IDF Retriever] -> Fetches Full Long Document
                                         |
                                         V
                                  [Text Chunker] -> Splits Doc into Chunks 1, 2, 3...
                                         |
                                         V
      (Map Step) -> [Fine-Tuned LLM] -> Summarizes Each Chunk -> [Summary 1, Summary 2, Summary 3...]
                                         |
                                         V
                                 [Combiner] -> Creates a single text from all chunk summaries
                                         |
                                         V
     (Reduce Step) -> [Fine-Tuned LLM] -> Creates a final, consolidated summary
                                         |
                                         V
                                   Final Output
```

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.10+
* Git
* A Hugging Face account with an access token.
* An NVIDIA GPU with at least 6GB VRAM is recommended for fine-tuning.

### 1. Setup

First, clone the repository and navigate into the project directory.

```bash
git clone [https://github.com/KS-Mohit/Legal-Doc-Summarization-LLM-QLoRA-RAG.git](https://github.com/KS-Mohit/Legal-Doc-Summarization-LLM-QLoRA-RAG.git)
cd Legal-Doc-Summarization-LLM-QLoRA-RAG
```

### 2. Create a Virtual Environment

Create and activate a virtual environment. This project uses `.venv310` as the standard name.

```bash
# Create the virtual environment
python -m venv .venv310

# Activate it (on Windows PowerShell)
.\.venv310\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 4. Hugging Face Authentication

Log in to your Hugging Face account to download the base model. You can also create a `.env` file in the project root with the line `HF_TOKEN="your_hugging_face_token"` for the scripts to automatically log in.

```bash
huggingface-cli login
```

## Usage Workflow

Follow these steps in order to prepare the data, fine-tune the model, and run the final application.

### Step 1: Data Preparation

1.  **Download the Dataset:** Download the legal document dataset from [Zenodo](https://zenodo.org/records/10056976).
2.  **Extract Data:** Extract the contents into the `datasets/` folder. Your directory structure should look like this:
    ```
    Legal-Doc-Summarization-LLM-QLoRA-RAG/
    └── datasets/
        └── IN-Ext/
            ├── judgement/
            └── summary/
    ```
3.  **Run Preprocessing Script:** Execute the script to format the raw data into a single training file using the native TinyLlama chat format.

    ```bash
    python src/data_preprocessing.py
    ```
    This will create a new file: `datasets/train_dataset.jsonl`.

### Step 2: Fine-Tuning the Model

Run the fine-tuning script. This will load the base TinyLlama model, apply QLoRA, and fine-tune it on your prepared dataset.

```bash
python src/finetune_llama.py
```

This process will take time depending on your GPU. Upon completion, it will save the final, trained LoRA adapter and tokenizer to a new folder named `final_adapter_model/`.

### Step 3: Run the RAG Pipeline

Now you can run the main application. This script will load your fine-tuned model and start an interactive session where you can query your legal documents.

```bash
python src/rag_pipeline.py
```

The application will prompt you to enter a query. Type in a case name or relevant keywords and the system will retrieve the document and generate a comprehensive summary using the Map-Reduce strategy. Type `exit` to quit.

## Code Structure

* `src/data_preprocessing.py`: Loads the raw `.txt` files, cleans them, and formats them into a `.jsonl` file with the correct TinyLlama prompt structure.
* `src/finetune_llama.py`: Handles the QLoRA fine-tuning process. It loads the base model, applies adapters, and trains the model on the preprocessed data, saving the final adapter.
* `src/rag_pipeline.py`: The main application. It sets up the retrieval system, loads the fine-tuned model, and implements the Map-Reduce logic for interactive summarization.
* `src/api.py`: A basic FastAPI endpoint for serving the model (optional).

## Future Work

This project provides a strong foundation that can be extended in several ways:

* **Improve Retrieval:** Replace the TF-IDF retriever with a more advanced vector-embedding model (e.g., Sentence-Transformers) for better semantic search.
* **Scale the Model:** Fine-tune a larger base model (e.g., Mistral-7B, Llama-3-8B) on a cloud GPU service to achieve higher-quality and more nuanced summaries.
* **Build a User Interface:** Wrap the RAG pipeline in a simple Gradio or Flask web interface for easier use.
* **Quantitative Evaluation:** Implement metrics like ROUGE to formally evaluate the quality of the generated summaries against the reference summaries.
