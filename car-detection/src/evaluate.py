"""
Evaluate YOLO model for car detection
"""

import os
import json
import argparse
from typing import Dict
from ultralytics import YOLO
import matplotlib.pyplot as plt


def evaluate_model(
    model_path: str,
    data_config: str,
    output_dir: str = "evaluation_results"
) -> Dict:
    """
    Evaluate YOLO model on test/validation set.
    
    Args:
        model_path: Path to trained YOLO model
        data_config: Path to data configuration YAML
        output_dir: Directory to save results
        
    Returns:
        Evaluation metrics dictionary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    print("Evaluating model...")
    metrics = model.val(data=data_config, save=True, plots=True)
    
    # Extract metrics
    results = {
        'model': model_path,
        'data_config': data_config,
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
        'fitness': float(metrics.fitness),
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"mAP50:    {results['map50']:.4f}")
    print(f"mAP50-95: {results['map50_95']:.4f}")
    print(f"Fitness:  {results['fitness']:.4f}")
    print("="*50)
    
    # Save results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def plot_training_curves(
    runs_dir: str,
    output_dir: str = "evaluation_results"
) -> None:
    """
    Plot training curves from training results.
    
    Args:
        runs_dir: Directory containing training runs
        output_dir: Directory to save plots
    """
    try:
        # Try to load and plot results
        results_csv = os.path.join(runs_dir, 'results.csv')
        
        if not os.path.exists(results_csv):
            print(f"Results file not found: {results_csv}")
            return
        
        import pandas as pd
        
        df = pd.read_csv(results_csv)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot loss curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if 'train/loss' in df.columns:
            axes[0, 0].plot(df['train/loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
        
        if 'val/loss' in df.columns:
            axes[0, 1].plot(df['val/loss'])
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
        
        if 'metrics/mAP50' in df.columns:
            axes[1, 0].plot(df['metrics/mAP50'])
            axes[1, 0].set_title('mAP50')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP50')
        
        if 'metrics/mAP50-95' in df.columns:
            axes[1, 1].plot(df['metrics/mAP50-95'])
            axes[1, 1].set_title('mAP50-95')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP50-95')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(output_path, dpi=100)
        print(f"Training curves saved to: {output_path}")
        
    except ImportError:
        print("pandas is required for plotting. Install with: pip install pandas")
    except Exception as e:
        print(f"Error plotting results: {e}")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate YOLO car detection model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model')
    parser.add_argument('--data', type=str, default='config/car_data.yaml',
                       help='Path to data configuration YAML')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Path to save evaluation results')
    parser.add_argument('--plot-training', type=str,
                       help='Path to training runs directory to plot training curves')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    # Check if data config exists
    if not os.path.exists(args.data):
        print(f"Error: Data config not found: {args.data}")
        return
    
    # Evaluate model
    evaluate_model(args.model, args.data, args.output)
    
    # Plot training curves if specified
    if args.plot_training:
        plot_training_curves(args.plot_training, args.output)


if __name__ == '__main__':
    main()
