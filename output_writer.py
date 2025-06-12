# output_writer.py
import os
import json
from typing import Any, List
from dataclasses import asdict, is_dataclass
from loguru import logger

def _save_json(data: Any, filepath: str, description: str):
    """Core JSON saving function that handles dataclasses."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return

    try:
        # Convert list of dataclasses to list of dictionaries
        if isinstance(data, list) and data and is_dataclass(data[0]):
            dict_data = [asdict(item) for item in data]
        else:
            dict_data = data

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dict_data, f, indent=4)
        logger.info(f"{description.capitalize()} saved to {filepath}")
    except IOError as e:
        logger.error(f"Error writing {description} to {filepath}: {e}")
    except TypeError as te:
        logger.critical(f"CRITICAL JSON SERIALIZATION ERROR in {description} for {filepath}: {te}. Data preview: {str(data)[:200]}...")

def save_final_output(filepath: str, data: List[Any]):
    """Saves the primary, simplified output file."""
    _save_json(data, filepath, "final output")

def save_processing_log(filepath: str, data: List[Any]):
    """Saves the detailed processing log file."""
    _save_json(data, filepath, "processing log")

def save_detailed_llm_report(filepath: str, data: List[Any]):
    """Saves the detailed LLM performance report."""
    _save_json(data, filepath, "LLM report")

def save_classification_report(filepath: str, report_content: str):
    """Saves the aggregate classification metrics report to a text file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"Classification report saved to {filepath}")
    except IOError as e:
        logger.error(f"Error writing classification report to {filepath}: {e}")
