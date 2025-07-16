"""Module for creating evaluation metric visualizations."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def create_metrics_bar_chart(results: pd.DataFrame, output_path: Path):
    """Create a bar chart comparing precision, recall, and F1 scores."""
    # Save data to CSV
    csv_path = output_path.with_suffix('.csv')
    results[['field', 'precision', 'recall', 'f1']].to_csv(csv_path, index=False)
    
    plt.figure(figsize=(12, 6))
    x = range(len(results['field']))
    width = 0.25
    
    plt.bar(x, results['precision'], width, label='Precision', color='skyblue')
    plt.bar([i + width for i in x], results['recall'], width, label='Recall', color='lightgreen')
    plt.bar([i + width * 2 for i in x], results['f1'], width, label='F1', color='salmon')
    
    plt.xlabel('Fields')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Scores by Field')
    plt.xticks([i + width for i in x], results['field'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()    
    plt.savefig(str(output_path))
    plt.close()

def create_metrics_heatmap(results: pd.DataFrame, output_path: Path):
    """Create a heatmap of the metrics."""
    metrics_data = results[['field', 'precision', 'recall', 'f1']].set_index('field')
    
    # Sort by F1 score in descending order
    metrics_data = metrics_data.sort_values('f1', ascending=False)
    
    # Save data to CSV
    csv_path = output_path.with_suffix('.csv')
    metrics_data.to_csv(csv_path)
    
    plt.figure(figsize=(7, 5))  # Slightly reduced figure size
    custom_cmap = sns.color_palette("RdYlGn", as_cmap=True)  # Red to Yellow to Green
    sns.heatmap(metrics_data, annot=True, cmap=custom_cmap, fmt='.3f', 
                vmin=0.0, vmax=1.0, center=0.3,
                annot_kws={'size': 12},  # Increased font size for cell values
                cbar_kws={'label': 'Score'},
                square=True)  # Make cells square and more compact
    plt.title('Metrics Heatmap', fontsize=14, pad=10)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=12)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Adjust layout to remove excess margins
    plt.tight_layout(pad=1.0)
    
    # Save with minimal borders
    plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_error_analysis_chart(results: pd.DataFrame, output_path: Path):
    """Create a stacked bar chart showing TP, FP, and FN."""
    error_data = results[['field', 'true_positives', 'false_positives', 'false_negatives']]
    
    # Save data to CSV
    csv_path = output_path.with_suffix('.csv')
    error_data.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(error_data))
    
    for column in ['true_positives', 'false_positives', 'false_negatives']:
        plt.bar(error_data['field'], error_data[column], bottom=bottom, label=column.replace('_', ' ').title())
        bottom += error_data[column]
    
    plt.xlabel('Fields')
    plt.ylabel('Count')
    plt.title('Error Analysis by Field')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()    
    plt.savefig(str(output_path))
    plt.close()

def load_model_results(results_folder: Path) -> dict:
    """Load evaluation results for all models."""
    model_results = {}
    for csv_file in results_folder.glob("evaluation_results_*.csv"):
        if "_cost" in csv_file.stem:
            continue
        model_name = csv_file.stem.replace("evaluation_results_", "")
        print(csv_file)
        model_results[model_name] = pd.read_csv(csv_file)
    return model_results

def create_model_comparison_chart(model_results: dict, output_path: Path):
    """Create a grouped bar chart comparing F1 scores across models."""
    # Get unique fields across all models
    all_fields = set()
    for results in model_results.values():
        all_fields.update(results['field'])
    all_fields = sorted(list(all_fields))
    
    # Prepare data for CSV
    comparison_data = []
    for model_name, results in model_results.items():
        model_f1 = {field: results[results['field'] == field]['f1'].iloc[0] if field in results['field'].values else 0 
                   for field in all_fields}
        model_f1['model'] = model_name
        comparison_data.append(model_f1)
    
    # Save data to CSV
    csv_path = output_path.with_suffix('.csv')
    pd.DataFrame(comparison_data).to_csv(csv_path, index=False)
    
    plt.figure(figsize=(15, 8))
    
    # Set up the positions for bars
    num_models = len(model_results)
    width = 0.8 / num_models
    positions = np.arange(len(all_fields))
    
    # Plot bars for each model
    for i, (model_name, results) in enumerate(model_results.items()):
        model_f1 = [results[results['field'] == field]['f1'].iloc[0] if field in results['field'].values else 0 
                   for field in all_fields]
        plt.bar(positions + i * width - (0.4 - width/2), 
               model_f1, 
               width, 
               label=model_name,
               alpha=0.8)
    
    plt.xlabel('Fields')
    plt.ylabel('F1 Score')
    plt.title('Model Comparison - F1 Scores by Field')
    plt.xticks(positions, all_fields, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches='tight')
    plt.close()

def create_model_comparison_heatmap(model_results: dict, output_path: Path):
    """Create a heatmap comparing models across metrics."""
    # Prepare data for heatmap
    heatmap_data = []
    for model_name, results in model_results.items():
        # Calculate mean scores for numeric columns only
        numeric_metrics = results[['precision', 'recall', 'f1']].astype(float)
        model_metrics = numeric_metrics.mean()
        heatmap_data.append(model_metrics)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=model_results.keys(),
                             columns=['precision', 'recall', 'f1'])
    
    # Sort by F1 score in descending order
    heatmap_df = heatmap_df.sort_values('f1', ascending=False)
    
    # Save data to CSV
    csv_path = output_path.with_suffix('.csv')
    heatmap_df.to_csv(csv_path)
    
    plt.figure(figsize=(8, len(model_results) * 0.6 + 1.5))  # Adjusted figure size ratio
    custom_cmap = sns.color_palette("RdYlGn", as_cmap=True)  # Red to Yellow to Green
    sns.heatmap(heatmap_df, annot=True, cmap=custom_cmap, fmt='.3f',
                vmin=0.0, vmax=1.0, center=0.5,
                annot_kws={'size': 12},  # Increased font size for cell values
                cbar_kws={'label': 'Score'},
                square=True)  # Make cells square and more compact
    plt.title('Model Comparison - Average Metrics', fontsize=14, pad=10)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=12)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Adjust layout to remove excess margins
    plt.tight_layout(pad=1.0)
    
    # Save with minimal borders
    plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0.3)
    plt.close()

def create_cost_bar_chart(cost_results: dict, output_path: Path):
    """Create a horizontal bar chart comparing costs per text across models."""
    # Calculate cost per text file for each model
    costs_per_text = {}
    colors = []
    
    for model, results in cost_results.items():
        cost_per_text = results['total_cost'].sum() / len(results)
        costs_per_text[model] = cost_per_text
        colors.append('lightgreen' if 'Discounted' in model else 'lightblue')
    
    # Sort models by cost in descending order
    costs_per_text = dict(sorted(costs_per_text.items(), key=lambda x: x[1], reverse=True))
    colors = ['lightgreen' if 'Discounted' in model else 'lightblue' for model in costs_per_text.keys()]
    
    # Save data to CSV
    csv_path = output_path.with_suffix('.csv')
    pd.DataFrame({
        'model': list(costs_per_text.keys()),
        'cost_per_text': list(costs_per_text.values()),
        'pricing_type': ['Discounted' if 'Discounted' in model else 'Regular' for model in costs_per_text.keys()]
    }).to_csv(csv_path, index=False)
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, max(6, len(costs_per_text) * 0.4)))
    y = range(len(costs_per_text))
    bars = plt.barh(y, list(costs_per_text.values()), color=colors)
    
    plt.ylabel('Models', fontsize=14)
    plt.xlabel('Cost per Text ($)', fontsize=14)
    plt.title('Cost per Text Comparison Across Models\n(Regular vs Discounted Pricing)', fontsize=16, pad=20)
    plt.yticks(y, list(costs_per_text.keys()), ha='right', fontsize=12)
    plt.xticks(fontsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Regular Pricing'),
        Patch(facecolor='lightgreen', label='Discounted Pricing')
    ]
    plt.legend(handles=legend_elements, fontsize=12)
    
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()

def create_cost_vs_metrics_scatter(model_results: dict, cost_results: dict, output_path: Path):
    """Create a horizontal bar chart comparing models by combined F1 and cost score."""
    data = []
    
    # Step 1: Gather raw data
    for model_name, results in cost_results.items():
        base_model = model_name.replace(" (Discounted)", "")
        if base_model not in model_results:
            continue

        metrics = model_results[base_model][['precision', 'recall', 'f1']].mean()
        cost_per_text = results['total_cost'].sum()

        data.append({
            'model': model_name,
            'cost': cost_per_text,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'pricing_type': results['pricing_type'].iloc[0]
        })

    df = pd.DataFrame(data)
    print(df)

    if df.empty:
        print("No data to plot.")
        return

    # Step 2: Normalize cost (lower cost = higher normalized value)
    min_cost = df['cost'].min()
    max_cost = df['cost'].max()
    df['cost_norm'] = (max_cost - df['cost']) / (max_cost - min_cost) if max_cost != min_cost else 1.0

    # Step 3: Combine f1 and cost_norm into a Harmonic Mean score
    print(df['f1'], df['cost_norm'])
    df['score'] = 2 * (df['f1'] * df['cost_norm']) / (df['f1'] + df['cost_norm'])

    # Step 4: Save CSV and sort
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    df = df.sort_values('score', ascending=True)

    # Step 5: Plot
    plt.figure(figsize=(10, max(6, len(df) * 0.4)))
    y = range(len(df))
    colors = ['lightblue' if pt == 'regular' else 'lightgreen' for pt in df['pricing_type']]
    bars = plt.barh(y, df['score'], color=colors)

    plt.ylabel('Models', fontsize=14)
    plt.xlabel('Score (Harmonic Mean of F1 and Normalized Cost)', fontsize=14)
    plt.title('Model Score', fontsize=16, pad=20)
    plt.yticks(y, df['model'], ha='right', fontsize=12)
    plt.xticks(fontsize=12)

    plt.legend(handles=[
        Patch(facecolor='lightblue', label='Regular Pricing'),
        Patch(facecolor='lightgreen', label='Discounted Pricing')
    ], fontsize=12)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2.,
                 f'{width:.3f}', ha='left', va='center', fontsize=11)

    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def load_cost_results(results_folder: Path) -> dict:
    """Load cost results for all models, including both regular and discount pricing."""
    cost_results = {}
    
    # Load both regular and discount cost files
    for csv_file in results_folder.glob("evaluation_results_*_cost.csv"):
        if "_model_discount_cost" in csv_file.stem:
            # Handle discount pricing files
            model_name = csv_file.stem.replace("evaluation_results_", "").replace("_model_discount_cost", "")
            df = pd.read_csv(csv_file)
            df['pricing_type'] = 'discounted'
            cost_results[f"{model_name} (Discounted)"] = df
        else:
            # Handle regular pricing files
            model_name = csv_file.stem.replace("evaluation_results_", "").replace("_cost", "")
            df = pd.read_csv(csv_file)
            df['pricing_type'] = 'regular'
            cost_results[model_name] = df
    
    return cost_results

def main():
    """Generate all visualizations."""
    base_dir = Path(__file__).parent
    output_dir = base_dir / "visualization_results"
    output_dir.mkdir(exist_ok=True)
    
    # Load results for all models
    model_results = load_model_results(base_dir)
    cost_results = load_cost_results(base_dir)  # Load cost results
    
    # Create comparison visualizations
    create_model_comparison_chart(model_results, output_dir / "model_comparison_f1.png")
    create_model_comparison_heatmap(model_results, output_dir / "model_comparison_heatmap.png")
    
    # Create cost visualizations if cost data is available
    if cost_results:
        create_cost_bar_chart(cost_results, output_dir / "model_cost_comparison.png")
        create_cost_vs_metrics_scatter(model_results, cost_results, output_dir / "cost_vs_metrics.png")
    
    # Create individual model visualizations
    for model_name, results in model_results.items():
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        create_metrics_bar_chart(results, model_dir / "metrics_bar_chart.png")
        create_metrics_heatmap(results, model_dir / "metrics_heatmap.png")
        create_error_analysis_chart(results, model_dir / "error_analysis.png")

if __name__ == "__main__":
    main()
