import glob
import json
import logging
import uuid
import argparse
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from adapters import DeepseekJSONAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluate_extraction import evaluate_metrics
from signatures import (
    ParcelSignature,
    ParcelDocumentation,
    ParcelsListSignature,
    ParcelDocumentationReferences,
)
from modules import ActionsAndEntities

import dspy


def process_file_wrapper(module, txt_path: str, model: str) -> str:
    """Process a single text file and save the extracted data.
    
    Args:
        module: The DSPy module instance to use for processing
        txt_path: Path to the input text file
        model: Name of the model being used
        
    Returns:
        str: Status message indicating success or that file was skipped
    """
    try:
        txt_path = Path(txt_path)
        output_path = txt_path.parent / f"{txt_path.stem}_{model.replace('/', '_')}.json"
        usage_path = txt_path.parent / f"{txt_path.stem}_{model.replace('/', '_')}_usage.json"
        
        if output_path.exists():
            return f"Skipped {txt_path} - output {output_path} already exists"

        # Read the text file
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract data using the module
        extracted_data = module(text)

        # Save extracted data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([model.model_dump() for model in extracted_data.entities], f, indent=2, ensure_ascii=False)

        with open(usage_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data.get_lm_usage(), f, indent=2, ensure_ascii=False)
        
        return f"Successfully processed {txt_path}"
    except Exception as e:
        logging.exception(f"Error processing {txt_path}: {str(e)}")
        return



def main():
    parser = argparse.ArgumentParser(description='Train and evaluate the model with configurable paths and limits.')
    parser.add_argument('--eval-path', type=str, default="../_results_for_evaluation/",
                      help='Path to evaluation examples directory')
    parser.add_argument('--dev-limit', type=int, default=50,
                      help='Maximum number of examples to use from evaluation set')
    parser.add_argument('--model', type=str, default="openai/gpt-4o-mini",
                      help='Model to use for training')
    
    args = parser.parse_args()
    
    # Initialize DSPy once
    model = args.model
    lm = dspy.LM(model, cache=False,)
    if 'deepseek' in model:
        dspy.settings.configure(lm=lm, adapter=(
            DeepseekJSONAdapter()
        ), track_usage=True)
    else:
        dspy.settings.configure(lm=lm, track_usage=True)
    
    dspy.settings.configure(track_usage=True)
    
    # Create module instance
    module = ActionsAndEntities()
    
    base_dir = Path(args.eval_path)
    txt_files = list(glob.iglob(str(base_dir / "*.txt"), recursive=False))[:args.dev_limit]
    process_args = [(txt_path, model) for txt_path in txt_files]
    
    # Use more threads since they're lightweight
    max_workers = 8
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(executor.submit(process_file_wrapper, module, *args) for args in process_args)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            logging.info(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
