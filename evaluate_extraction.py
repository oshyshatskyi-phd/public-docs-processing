import json
from pathlib import Path
from typing import Dict
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Price per 1M tokens for different models
pricing = {
    "openai/gpt-4o": {"input": 2.5, "output": 10.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "openai/gpt-4.1-2025-04-14": {"input": 2.0, "output": 8.0},
    "openai/gpt-4.1-mini-2025-04-14": {"input": 0.4, "output": 1.6},
    "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini/gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 0.6},
    "xai/grok-3-latest": {"input": 3.0, "output": 15.0},
    "xai/grok-3-mini-beta": {"input": 0.3, "output": 0.5},
    "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
}

discount_pricing = {
    "deepseek/deepseek-chat": {"input": 0.135, "output": 0.550},
    "openai/gpt-4o": {"input": 1.25, "output": 5.0},
    "openai/gpt-4o-mini": {"input": 0.075, "output": 0.3},
    "openai/gpt-4.1-2025-04-14": {"input": 1.0, "output": 4.0},
    "openai/gpt-4.1-mini-2025-04-14": {"input": 0.2, "output": 0.8},
    "gemini/gemini-2.5-flash-preview-05-20": {"input": 0.075, "output": 0.3},
}

def load_json(json_path: Path) -> Dict:
    """Load a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_parcels(human_data: Dict, ai_data: Dict) -> Dict[str, Dict[str, int]]:
    """Compare individual fields between human and AI data."""
    metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    # Fields to evaluate
    fields = ["number", "area", "area_unit", "purpose_code", "category", "ownership"]
    
    # Match parcels based on cadastral numbers when available
    human_parcels = {p.get("number"): p for p in human_data if p.get("number")}
    ai_parcels = {p.get("number"): p for p in ai_data if p.get("number")}
    
    logger.info(f"\nComparing parcels:")
    logger.info(f"Found {len(human_parcels)} human parcels and {len(ai_parcels)} AI parcels")
    
    # Compare matched parcels
    for number, human_parcel in human_parcels.items():
        ai_parcel = ai_parcels.get(number)

        if ai_parcel:
            logger.info(f"\nComparing parcel {number}:")
            for field in fields:
                human_value = human_parcel.get(field)
                ai_value = ai_parcel.get(field)
                
                if human_value == ai_value:
                    metrics[field]["tp"] += 1
                    logger.info(f"  ✓ {field}: {human_value}")
                elif human_value and ai_value is None:
                    metrics[field]["fn"] += 1
                    logger.info(f"  × {field}: human={human_value}, AI=missing")
                elif human_value is None and ai_value:
                    metrics[field]["fp"] += 1
                    logger.info(f"  × {field}: human=missing, AI={ai_value}")
                else:
                    metrics[field]["fn"] += 1
                    metrics[field]["fp"] += 1
                    logger.info(f"  × {field}: human={human_value}, AI={ai_value}")
        else:
            # If AI didn't find the parcel, count as false negative for all fields
            logger.warning(f"\nMissing parcel {number} in AI data")
            for field in fields:
                if human_parcel.get(field):
                    metrics[field]["fn"] += 1
                    logger.warning(f"  × {field}: human={human_parcel.get(field)}, AI=missing")
    
    # Count unmatched AI parcels as false positives for all fields
    for number, ai_parcel in ai_parcels.items():
        if number in human_parcels:
            continue

        logger.warning(f"\nExtra parcel {number} in AI data")
        for field in fields:
            if ai_parcel.get(field):
                metrics[field]["fp"] += 1
                logger.warning(f"  × {field}: human=missing, AI={ai_parcel.get(field)}")
    
    return metrics

def compare_documentation(human_data: Dict, ai_data: Dict) -> Dict[str, Dict[str, int]]:
    """Compare documentation types and their parcel references."""
    doc_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    def get_doc_parcel_tuples(data: Dict) -> set:
        """Extract set of (doc_type, parcel_number) tuples from data"""
        result = set()

        print(data)
        
        for doc in data:
            if doc["type"] != "documentation":
                continue
            doc_type = doc["documentation_type"]
            result.add((doc_type, tuple(doc.get("involved_parcels", tuple()))))

        print(result)
        return result
    
    # Get sets of (doc_type, parcel_number) tuples
    human_tuples = get_doc_parcel_tuples(human_data)
    ai_tuples = get_doc_parcel_tuples(ai_data)

    logger.info(f"\nComparing documentation:")
    logger.info(f"Found {len(human_tuples)} human doc-parcel pairs and {len(ai_tuples)} AI doc-parcel pairs")
    
    # Calculate metrics for parcel references
    matching_tuples = human_tuples & ai_tuples
    false_positives = ai_tuples - human_tuples
    false_negatives = human_tuples - ai_tuples
    
    doc_metrics["parcel_references"]["tp"] = len(matching_tuples)
    doc_metrics["parcel_references"]["fp"] = len(false_positives)
    doc_metrics["parcel_references"]["fn"] = len(false_negatives)
    
    if matching_tuples:
        logger.info("\nMatching document-parcel pairs:")
        for doc_type, parcel in matching_tuples:
            logger.info(f"  ✓ {doc_type} - Parcel {parcel}")
    
    if false_negatives:
        logger.warning("\nMissing document-parcel pairs (in human but not in AI):")
        for doc_type, parcel in false_negatives:
            logger.warning(f"  × {doc_type} - Parcel {parcel}")
    
    if false_positives:
        logger.warning("\nExtra document-parcel pairs (in AI but not in human):")
        for doc_type, parcel in false_positives:
            logger.warning(f"  × {doc_type} - Parcel {parcel}")
    
    # Calculate metrics for document types using both sets and frequency counts
    from collections import Counter
    
    # Keep sets for logging unique types
    human_doc_types = {t[0] for t in human_tuples}
    ai_doc_types = {t[0] for t in ai_tuples}
    
    matching_types = human_doc_types & ai_doc_types
    missing_types = human_doc_types - ai_doc_types
    extra_types = ai_doc_types - human_doc_types

    # Use counters for frequency-based metrics
    human_type_counts = Counter(t[0] for t in human_tuples)
    ai_type_counts = Counter(t[0] for t in ai_tuples)

    logger.info(f"\nComparing document types:")
    logger.info(f"Found {len(human_doc_types)} human doc types and {len(ai_doc_types)} AI doc types")
    logger.debug(f"Matching types: {len(matching_types)}, Missing types: {len(missing_types)}, Extra types: {len(extra_types)}")
    logger.info(f"Human types: {human_type_counts}")
    logger.info(f"AI types: {ai_type_counts}")
    
    # True positives: minimum frequency between human and AI for each type
    tp = sum(min(human_type_counts[t], ai_type_counts[t]) for t in matching_types)
    
    # False positives: extra occurrences in AI predictions
    fp = sum(max(0, ai_type_counts[t] - human_type_counts[t]) for t in ai_type_counts)
    
    # False negatives: missing occurrences from human annotations
    fn = sum(max(0, human_type_counts[t] - ai_type_counts[t]) for t in human_type_counts)
    
    doc_metrics["documentation_type"]["tp"] = tp
    doc_metrics["documentation_type"]["fp"] = fp
    doc_metrics["documentation_type"]["fn"] = fn
    
    logger.info("\nDocument types comparison:")
    if matching_types:
        logger.info("Matching types:")
        for doc_type in matching_types:
            logger.info(f"  ✓ {doc_type} (Human: {human_type_counts[doc_type]}, AI: {ai_type_counts[doc_type]})")
    
    if missing_types:
        logger.warning("Missing types (in human but not in AI):")
        for doc_type in missing_types:
            logger.warning(f"  × {doc_type} (Human: {human_type_counts[doc_type]}, AI: 0)")
    
    if extra_types:
        logger.warning("Extra types (in AI but not in human):")
        for doc_type in extra_types:
            logger.warning(f"  × {doc_type} (Human: 0, AI: {ai_type_counts[doc_type]})")
    
    return doc_metrics

def calculate_metrics(stats: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Calculate precision, recall, and F1 score for each field."""
    results = []
    
    for field, values in stats.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "field": field,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        })
    
    return pd.DataFrame(results)

def evaluate_metrics(human_json: Path, ai_json: Path) -> float:
    """Calculate the average F1 score across all fields."""
    all_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    process_file_metrics(human_json, ai_json, all_metrics)

    f1_scores = []
    for field, values in all_metrics.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def process_file_metrics(human_json: object, ai_json: object, all_metrics: Dict) -> None:
    """Process metrics for a single file pair and update the aggregate metrics."""
    human_data = human_json
    ai_data = ai_json

    print(f"Human data: {len(human_data)} parcels, AI data: {len(ai_data)} parcels")
      # Compare parcel fields and aggregate metrics
    parcel_metrics = compare_parcels(human_data, ai_data)
    for field, values in parcel_metrics.items():
        for k, v in values.items():
            all_metrics[field][k] += v
    
    # Compare documentation and aggregate metrics
    doc_metrics = compare_documentation(human_data, ai_data)
    for field, values in doc_metrics.items():
        for k, v in values.items():
            all_metrics[field][k] += v

def evaluate_model(txt_folder: Path, model_name: str) -> pd.DataFrame:
    """Evaluate model performance against human validation."""
    all_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    # Find all text files with both human and AI annotations
    for txt_path in txt_folder.glob("*.txt"):
        print(f"Processing {txt_path.name}...")
        human_json_path = txt_path.with_name(f"{txt_path.stem}_human.json")
        ai_json_path = txt_path.with_name(f"{txt_path.stem}_{model_name}.json")

        if not human_json_path.exists():
            continue
        
        human_json = load_json(human_json_path)

        if not ai_json_path.exists():
            ai_json = []
        else:
            ai_json = load_json(ai_json_path)
        
        process_file_metrics(human_json, ai_json, all_metrics) 
    
    return calculate_metrics(all_metrics)

def calculate_model_costs(model_name: str, txt_folder: Path, use_discount: bool = False) -> pd.DataFrame:
    """Calculate the costs for model usage based on token counts."""
    costs = []
    
    # Find all usage files for the model
    for txt_path in txt_folder.glob("*.txt"):
        usage_path = txt_path.with_name(f"{txt_path.stem}_{model_name}_usage.json")
        if not usage_path.exists():
            continue

        with open(usage_path, 'r') as f:
            usage_data = json.load(f)

        for pricing_model, model_data in usage_data.items():
            prompt_tokens = model_data.get("prompt_tokens", 0)
            completion_tokens = model_data.get("completion_tokens", 0)
            
            # Select pricing based on whether to use discount
            if use_discount and pricing_model in discount_pricing:
                model_pricing = discount_pricing[pricing_model]
            else:
                model_pricing = pricing[pricing_model]
            
            input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
            output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]
            total_cost = input_cost + output_cost
            
            costs.append({
                "file": txt_path.stem,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "pricing_type": "discount" if use_discount else "regular"
            })
    
    if not costs:
        return pd.DataFrame()
    
    return pd.DataFrame(costs)

def main():
    parser = argparse.ArgumentParser(description='Evaluate AI model extraction against human validation.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing text and JSON files')
    parser.add_argument('--model', type=str, default="deepseek_deepseek-chat", help='Model identifier (default: deepseek_chat)')
    args = parser.parse_args()
    
    base_dir = Path(args.folder_path)
    if not base_dir.exists():
        print(f"Error: Folder {base_dir} does not exist")
        return
    
    # Evaluate model performance
    results = evaluate_model(base_dir, args.model)
    
    # Calculate regular cost metrics
    cost_results = calculate_model_costs(args.model, base_dir, use_discount=False)
    
    # Calculate discount cost metrics if applicable
    model_base_name = args.model.split("_")[-1]  # Extract the base model name
    full_model_name = f"{args.model.split('_')[0]}/{model_base_name}"
    if full_model_name in discount_pricing:
        discount_cost_results = calculate_model_costs(args.model, base_dir, use_discount=True)
    else:
        discount_cost_results = pd.DataFrame()
    
    # Print performance results
    print(f"\nEvaluation Results for {args.model}:")
    print(results.to_string(index=False))
    
    if not cost_results.empty:
        print(f"\nRegular Cost Analysis for {args.model}:")
        print(f"Total cost: ${cost_results['total_cost'].sum():.4f}")
        print(f"Average cost per file: ${cost_results['total_cost'].mean():.4f}")
        print(f"Total tokens: {cost_results['total_tokens'].sum()}")
        
        if not discount_cost_results.empty:
            print(f"\nDiscount Cost Analysis for {args.model}:")
            print(f"Total cost: ${discount_cost_results['total_cost'].sum():.4f}")
            print(f"Average cost per file: ${discount_cost_results['total_cost'].mean():.4f}")
            print(f"Total tokens: {discount_cost_results['total_tokens'].sum()}")
    
    # Save results to CSV
    metrics_file = f"evaluation_results_{args.model}.csv"
    results.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to {metrics_file}")
    
    if not cost_results.empty:
        # Save regular cost results
        cost_file = f"evaluation_results_{args.model}_cost.csv"
        cost_results.to_csv(cost_file, index=False)
        print(f"Regular cost analysis saved to {cost_file}")
        
        # Save discount cost results if available
        if not discount_cost_results.empty:
            discount_cost_file = f"evaluation_results_{args.model}_model_discount_cost.csv"
            discount_cost_results.to_csv(discount_cost_file, index=False)
            print(f"Discount cost analysis saved to {discount_cost_file}")

if __name__ == "__main__":
    main()
