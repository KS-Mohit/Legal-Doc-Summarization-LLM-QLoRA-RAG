# src/finetune_llama.py

import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dotenv import load_dotenv
from huggingface_hub import login

# --- Setup and Configuration ---
# Load environment variables from .env file
load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

# Use pathlib for robust path handling
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Dataset and Tokenizer Functions ---
def load_custom_dataset(jsonl_path):
    """Loads a dataset from a .jsonl file."""
    return load_dataset("json", data_files={"train": str(jsonl_path)})["train"]

def tokenize_function(example, tokenizer, max_length=2048):
    """Tokenizes a single example from the dataset."""
    # This tokenizes the 'text' field from your train_dataset.jsonl
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

def main():
    # --- Model and Path Configuration ---
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Correctly path to the dataset we created
    dataset_path = PROJECT_ROOT / "datasets" / "train_dataset.jsonl"
    
    # Define output directories clearly
    training_output_dir = PROJECT_ROOT / "training_checkpoints"
    final_model_dir = PROJECT_ROOT / "final_adapter_model"

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- QLoRA Configuration ---
    # 4-bit config tailored for consumer GPUs like yours
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # Correct for RTX 30-series
    )

    # --- Model Loading ---
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA Configuration ---
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    # --- Data Processing ---
    dataset = load_custom_dataset(dataset_path)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True,
        remove_columns=dataset.column_names # Keep only model inputs
    )

    # --- Training ---
    # Training arguments optimized for low-VRAM GPUs
    training_args = TrainingArguments(
        output_dir=str(training_output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # Effective batch size = 8
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch", # Saves a checkpoint after each epoch
        fp16=True, # Essential for mixed-precision training on your GPU
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("--- Starting Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")

    # --- Save Final Model ---
    # This now saves to a clean folder inside your project directory
    final_model_dir.mkdir(exist_ok=True)
    trainer.model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"✅ Final model adapter and tokenizer saved to: {final_model_dir}")

if __name__ == "__main__":
    main()