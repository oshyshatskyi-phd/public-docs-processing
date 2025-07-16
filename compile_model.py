import glob
import json
import logging
import uuid
import argparse
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from adapters import DeepseekJSONAdapter
from evaluate_extraction import evaluate_metrics
from signatures import (
    ParcelSignature,
    ParcelDocumentation,
    ParcelsListSignature,
    ParcelDocumentationReferences,
)
from modules import ActionsAndEntities

import dspy

# dspy.settings.configure(track_usage=True)

from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from dspy.teleprompt import MIPROv2


def process_file(txt_path: Path, model) -> Dict:
    """Process a single text file."""
    try:
        output_path = Path(txt_path).parent / f"{Path(txt_path).stem}_{model.replace('/', '_')}.json"
        
        if output_path.exists():
            return f"Skipped {txt_path} - output already exists"

        # Read the text file
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract data
        extracted_data = module(text)

        # Save extracted data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([model.model_dump() for model in sum(extracted_data, [])], f, indent=2, ensure_ascii=False)
        
        return f"Successfully processed {txt_path}"
    except Exception as e:
        return f"Error processing {txt_path}: {str(e)}"


def create_training_examples(path):
    """Create training examples from labeled data files."""
    training_examples = []
    base_dir = Path(path)
    
    # Find all JSON files that have corresponding text files
    json_files = glob.glob(str(base_dir / "*_human.json"), recursive=False)
    
    for json_path in json_files:
        txt_path = json_path.replace("_human.json", ".txt")
        if not Path(txt_path).exists():
            continue
            
        try:
            # Read the input text
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            # Read the expected output
            with open(json_path, "r", encoding="utf-8") as f:
                expected = json.load(f)

            example = dspy.Example(text=text, expected=expected).with_inputs("text")
            training_examples.append(example)
            
        except Exception as e:
            logging.warning(f"Error processing {json_path}: {str(e)}")
            continue
            
    return training_examples


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate the model with configurable paths and limits.')
    parser.add_argument('--train-path', type=str, default="../_results_for_learning/",
                      help='Path to training examples directory')
    parser.add_argument('--eval-path', type=str, default="../_results_for_evaluation/",
                      help='Path to evaluation examples directory')
    parser.add_argument('--dev-limit', type=int, default=50,
                      help='Maximum number of examples to use from evaluation set')
    parser.add_argument('--model', type=str, default="openai/gpt-4o-mini",
                      help='Model to use for training')
    
    args = parser.parse_args()
    
    # Initialize DSPy once
    model = args.model
    lm = dspy.LM(model)
    dspy.configure(lm=lm, adapter=(
        # DeepseekJSONAdapter()
    ),)
    
    # Create module instance
    global module
    module = ActionsAndEntities()
        
    # Get training examples
    trainset = create_training_examples(args.train_path)
    devset = trainset + create_training_examples(args.eval_path)[:args.dev_limit]

    def test_evaluate_metrics(example, pred, trace=None):
        """Custom evaluation function to compute metrics."""
        scote = evaluate_metrics(example.expected, [m.model_dump() for m in pred])
        logging.info("Score:", scote)
        return scote

    optimizer = MIPROv2(metric=test_evaluate_metrics)

    compiled_pipeline = optimizer.compile(
        ActionsAndEntities(),
        trainset=trainset,
        requires_permission_to_run=False
    )    
        
    logging.info(f"Evaluating the compiled and optimized pipeline ...")
    # Set up the `evaluate_on_hotpotqa` function. 
    evaluate_on_hotpotqa = Evaluate(
        devset=devset, 
        num_threads=1, 
        display_progress=True, 
        display_table=10)

    # Evaluate the compiled pipeline on the HotPotQA dataset
    compiled_pipeline_retrieval_score = evaluate_on_hotpotqa(
        compiled_pipeline, metric=test_evaluate_metrics)
    logging.info(f"## Retrieval Score for compiled pipeline: {compiled_pipeline_retrieval_score}")

    logging.info(f"Saving the optimized pipeline ....")
    compiled_pipeline.save("optimized_pipeline.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
