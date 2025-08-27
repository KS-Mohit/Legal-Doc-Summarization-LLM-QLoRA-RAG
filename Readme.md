# Hybrid Multi-Agent Legal AI Summarizer

This project presents a sophisticated, full-stack application for summarizing and querying long legal documents. It features a hybrid, multi-agent architecture that leverages a locally fine-tuned specialist model for summarization and a powerful, general-purpose API for conversational Q&A.

The system is built around two core agents:
1.  **A Summarizer Agent:** Powered by a locally fine-tuned **TinyLlama-1.1B** model using **QLoRA**. This agent is an expert at processing raw legal text and uses a **Map-Reduce** strategy to generate high-quality, comprehensive summaries of documents that exceed the model's context window.
2.  **A Q&A Agent:** Powered by the **Google Gemini API**. This agent takes the context from the Summarizer Agent and provides high-fidelity, conversational answers to follow-up questions, leveraging the reasoning capabilities of a state-of-the-art large language model.

The entire system is served through a Python/Flask backend and is controlled by a clean, interactive web UI that includes session management to switch between multiple summarized documents.

## ✨ Key Features

* **Hybrid AI Architecture:** Combines a local, fine-tuned specialist model (TinyLlama) for summarization with a powerful generalist model (Gemini) for advanced Q&A.
* **Multi-Agent System:** A robust backend orchestrates tasks between the Summarizer Agent and the Q&A Agent based on the conversational state.
* **Long Document Handling:** Implements a **Map-Reduce** strategy to process and generate comprehensive summaries for legal documents of any length.
* **Full-Stack Application:** Features a decoupled architecture with a Python/Flask backend API and a vanilla HTML/CSS/JS frontend.
* **Interactive UI with Session History:** The user interface allows for seamless conversation, streaming of summaries, and the ability to switch between the contexts of previously summarized documents.

## 🏛️ Architecture: The Hybrid Multi-Agent Workflow

```
+------------------------------------------------------------------+
|                      Frontend (index.html)                       |
+------------------------------------------------------------------+
      | (User Query)                                 ^ (Response)
      |                                              |
      V                                              |
+------------------------------------------------------------------+
|                     Backend API (app.py)                         |
|                                                                  |
|  (Orchestrator: Is there a summary context?)                     |
|      | (No)                                  | (Yes)             |
|      V                                       V                   |
|  Summarizer Agent (Local TinyLlama)        Q&A Agent (Gemini API)|
|  - Retrieve Document                       - Receive Question    |
|  - Map: Summarize Chunks                   - Receive Context     |
|  - Reduce: Consolidate Summary             - Call Gemini API     |
|      |                                       |                   |
|      +-----------------> (Update Context) <--+                   |
+------------------------------------------------------------------+
```

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.10+
* Git
* An NVIDIA GPU with at least 6GB VRAM (for running the local model).
* A **Hugging Face Account** with an access token.
* A **Google AI Studio API Key** for the Gemini API.

### 1. Setup

Clone the repository and navigate into the project directory.

```bash
git clone [https://github.com/KS-Mohit/Legal-Doc-Summarization-LLM-QLoRA-RAG.git](https://github.com/KS-Mohit/Legal-Doc-Summarization-LLM-QLoRA-RAG.git)
cd Legal-Doc-Summarization-LLM-QLoRA-RAG
```

### 2. Create a Virtual Environment & Install Dependencies

```bash
# Create the virtual environment
python -m venv .venv310

# Activate it (on Windows PowerShell)
.\.venv310\Scripts\Activate.ps1

# Install all required packages
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a file named `.env` in the root of your project folder. Add your API keys to this file:

```env
# .env file
HF_TOKEN="your_hugging_face_token_here"
GEMINI_API_KEY="your_google_ai_studio_api_key_here"
```

## ⚙️ Usage Workflow

Follow these steps in order to prepare the data, fine-tune the model, and run the final application.

### Step 1: Data Preparation

1.  **Download the Dataset:** Download the legal document dataset from [Zenodo](https://zenodo.org/records/10056976).
2.  **Extract Data:** Extract the contents into the `datasets/` folder.
3.  **Run Preprocessing Script:** Execute the script to format the raw data into a single training file using the native TinyLlama chat format.

    ```bash
    python backend/data_preprocessing.py
    ```
    This will create `datasets/train_dataset.jsonl`.

### Step 2: Fine-Tuning the Model

Run the fine-tuning script to train your specialist summarization model.

```bash
python backend/finetune_llama.py
```
This will save the final, trained LoRA adapter to the `final_adapter_model/` folder.

### Step 3: Run the Full-Stack Application

1.  **Start the Backend Server:**
    Open a terminal, activate your virtual environment, and run the Flask app. **Leave this terminal running.**
    ```bash
    python backend/app.py
    ```

2.  **Launch the Frontend UI:**
    Navigate to the project folder in your file explorer, open the `frontend` directory, and double-click the `index.html` file to open it in your browser.

You can now interact with the application. Start by summarizing a document, and then ask follow-up questions to the Gemini-powered chatbot.

## 📂 Project Structure

* `frontend/index.html`: The self-contained user interface for the application.
* `backend/app.py`: The Flask backend server that loads the models and orchestrates the multi-agent system.
* `backend/finetune_llama.py`: The script for QLoRA fine-tuning of the TinyLlama model.
* `backend/data_preprocessing.py`: The script for preparing the training data.
* `datasets/`: The folder for the raw and processed legal document data.
* `final_adapter_model/`: The output directory for the fine-tuned model adapter.

## 🙏 Acknowledgements

This project was developed by building upon the foundational concepts and structure provided by the original [Legal-Doc-Summarization-LLM-LoRA-RAG repository by aryanmangal769](https://github.com/aryanmangal769/Legal-Doc-Summarization-LLM-LoRA-RAG) and the UI from the [chat-with-jfk-files repository by voynow](https://github.com/voynow/chat-with-jfk-files).
