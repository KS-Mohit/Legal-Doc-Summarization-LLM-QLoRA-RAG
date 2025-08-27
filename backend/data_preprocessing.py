# src/data_preprocessing.py

import os
import json
import logging
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "IN-Ext"

# Updated CONFIG to process all data and save to a single, unified file.
CONFIG = {
    "judgement_dir": DATA_ROOT / "judgement",
    "full_summary_dir": DATA_ROOT / "summary" / "full",
    "segment_summary_dir": DATA_ROOT / "summary" / "segment-wise",
    "output_dir": PROJECT_ROOT / "datasets", # Save directly in the datasets folder
    "output_filename": "train_dataset.jsonl", # A single file for training
    "authors": ["A1", "A2"], # Process data from both authors
    "segments": ["analysis", "argument", "facts", "judgement", "statute"],
}

# --- Helper to safely read text files ---
def safe_read_text(path: Path) -> str:
    """Reads a text file, ignoring potential encoding errors."""
    return path.read_text(encoding="utf-8", errors="ignore")

# --- CORRECT TinyLlama Formatting ---
def format_for_tinyllama(judgement_text: str, summary_text: str) -> str:
    """
    Formats the input and output text into the specific prompt structure
    required by TinyLlama-Chat models.
    """
    return f"""<|system|>
You are an expert legal assistant. Your task is to provide a clear and concise summary of the provided legal document.</s>
<|user|>
Summarize the following legal document:
{judgement_text}</s>
<|assistant|>
{summary_text}</s>"""

# --- Main Preprocessing Logic ---
def process_data():
    """Loads, formats, and saves the dataset for fine-tuning."""
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    output_path = CONFIG["output_dir"] / CONFIG["output_filename"]
    processed_records = []

    for author in CONFIG["authors"]:
        logging.info(f"Processing data for author: {author}")

        # --- FULL Summaries ---
        author_full_summary_dir = CONFIG["full_summary_dir"] / author
        if not author_full_summary_dir.exists():
            logging.warning(f"Full summary directory not found for author {author}, skipping.")
        else:
            logging.info(f"Loading full summaries for {author}...")
            for summary_file in tqdm(list(author_full_summary_dir.glob("*.txt")), desc=f"Full - {author}"):
                judgement_file = CONFIG["judgement_dir"] / summary_file.name
                if judgement_file.exists():
                    try:
                        judgement_text = safe_read_text(judgement_file)
                        summary_text = safe_read_text(summary_file)
                        # Use the correct formatting function
                        formatted_text = format_for_tinyllama(judgement_text, summary_text)
                        processed_records.append({"text": formatted_text})
                    except Exception as e:
                        logging.error(f"Could not process file {summary_file.name}: {e}")

        # --- SEGMENT-WISE Summaries ---
        analysis_dir = CONFIG["segment_summary_dir"] / author / "analysis"
        if not analysis_dir.exists():
            logging.warning(f"Segment summary 'analysis' directory not found for author {author}, skipping segments.")
        else:
            logging.info(f"Loading segment-wise summaries for {author}...")
            for base_file in tqdm(list(analysis_dir.glob("*.txt")), desc=f"Segment - {author}"):
                judgement_file = CONFIG["judgement_dir"] / base_file.name
                if not judgement_file.exists():
                    continue

                try:
                    judgement_text = safe_read_text(judgement_file)
                    concatenated_summary = []
                    for segment in CONFIG["segments"]:
                        segment_file = CONFIG["segment_summary_dir"] / author / segment / base_file.name
                        if segment_file.exists():
                            segment_text = safe_read_text(segment_file)
                            concatenated_summary.append(segment_text)

                    if concatenated_summary:
                        full_segment_summary = "\n\n".join(concatenated_summary)
                        # Use the correct formatting function
                        formatted_text = format_for_tinyllama(judgement_text, full_segment_summary)
                        processed_records.append({"text": formatted_text})
                except Exception as e:
                    logging.error(f"Could not process segmented file {base_file.name}: {e}")

    if not processed_records:
        logging.error("No records were processed! Please double-check the contents of your dataset directories.")
        return

    logging.info(f"Saving {len(processed_records)} processed records to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in processed_records:
            f.write(json.dumps(record) + "\n")

    logging.info(f"Dataset preparation complete! File saved at {output_path}")


if __name__ == "__main__":
    process_data()