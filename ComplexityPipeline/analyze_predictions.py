#!/usr/bin/env python3
"""
Diagnostic script to analyze Easy→Medium misclassification patterns.
Helps identify why the model confuses Easy sequences with Medium ones.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple
import logging

# Import your model and pipeline
from model_architecture import MultimodalRNN
from pipeline.tasks import ModelPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize analyzer with model and config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.predictor = ModelPredictor(config_path)
        self.class_names = self.config['multimodal_model']['class_names']
        
    def analyze_feature_patterns(self, predictions_csv: str) -> Dict:
        """Analyze feature patterns for misclassified Easy→Medium cases."""
        
        # Load predictions with features
        df = pd.read_csv(predictions_csv)
        
        # Filter Easy sequences predicted as Medium
        easy_as_medium = df[(df['true_label'] == 'Easy') & (df['predicted_label'] == 'Medium')]
        correct_easy = df[(df['true_label'] == 'Easy') & (df['predicted_label'] == 'Easy')]
        correct_medium = df[(df['true_label'] == 'Medium') & (df['predicted_label'] == 'Medium')]
        
        logger.info(f"Easy→Medium misclassifications: {len(easy_as_medium)}")
        logger.info(f"Correct Easy predictions: {len(correct_easy)}")
        logger.info(f"Correct Medium predictions: {len(correct_medium)}")
        
        # Feature columns (static features)
        feature_cols = [col for col in df.columns if col.endswith('_score')]
        
        analysis = {
            'misclassification_count': len(easy_as_medium),
            'total_easy_samples': len(df[df['true_label'] == 'Easy']),
            'easy_to_medium_rate': len(easy_as_medium) / len(df[df['true_label'] == 'Easy']) if len(df[df['true_label'] == 'Easy']) > 0 else 0,
            'feature_analysis': {}
        }
        
        # Compare feature distributions
        for feature in feature_cols:
            if feature in df.columns:
                easy_misclass_mean = easy_as_medium[feature].mean() if len(easy_as_medium) > 0 else 0
                easy_correct_mean = correct_easy[feature].mean() if len(correct_easy) > 0 else 0
                medium_correct_mean = correct_medium[feature].mean() if len(correct_medium) > 0 else 0
                
                analysis['feature_analysis'][feature] = {
                    'easy_misclassified_mean': easy_misclass_mean,
                    'easy_correct_mean': easy_correct_mean,
                    'medium_correct_mean': medium_correct_mean,
                    'easy_misclass_closer_to_medium': abs(easy_misclass_mean - medium_correct_mean) < abs(easy_misclass_mean - easy_correct_mean)
                }
        
        return analysis
    
    def analyze_confidence_patterns(self, predictions_csv: str) -> Dict:
        """Analyze confidence patterns for Easy→Medium misclassifications."""
        
        df = pd.read_csv(predictions_csv)
        
        # Filter cases
        easy_as_medium = df[(df['true_label'] == 'Easy') & (df['predicted_label'] == 'Medium')]
        correct_easy = df[(df['true_label'] == 'Easy') & (df['predicted_label'] == 'Easy')]
        
        confidence_analysis = {
            'easy_misclass_confidence': {
                'mean': easy_as_medium['confidence'].mean() if len(easy_as_medium) > 0 else 0,
                'std': easy_as_medium['confidence'].std() if len(easy_as_medium) > 0 else 0,
                'median': easy_as_medium['confidence'].median() if len(easy_as_medium) > 0 else 0
            },
            'easy_correct_confidence': {
                'mean': correct_easy['confidence'].mean() if len(correct_easy) > 0 else 0,
                'std': correct_easy['confidence'].std() if len(correct_easy) > 0 else 0,
                'median': correct_easy['confidence'].median() if len(correct_easy) > 0 else 0
            }
        }
        
        return confidence_analysis
    
    def plot_confusion_analysis(self, predictions_csv: str, save_dir: str = "analysis_plots"):
        """Create visualizations for misclassification analysis."""
        
        Path(save_dir).mkdir(exist_ok=True)
        df = pd.read_csv(predictions_csv)
        
        # 1. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=self.class_names)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature distributions for Easy→Medium cases
        feature_cols = [col for col in df.columns if col.endswith('_score')]
        
        if feature_cols:
            easy_as_medium = df[(df['true_label'] == 'Easy') & (df['predicted_label'] == 'Medium')]
            correct_easy = df[(df['true_label'] == 'Easy') & (df['predicted_label'] == 'Easy')]
            correct_medium = df[(df['true_label'] == 'Medium') & (df['predicted_label'] == 'Medium')]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(feature_cols[:6]):  # Plot first 6 features
                if i < len(axes):
                    ax = axes[i]
                    
                    if len(easy_as_medium) > 0:
                        ax.hist(easy_as_medium[feature], alpha=0.7, label='Easy→Medium', bins=20, color='red')
                    if len(correct_easy) > 0:
                        ax.hist(correct_easy[feature], alpha=0.7, label='Easy (correct)', bins=20, color='green')
                    if len(correct_medium) > 0:
                        ax.hist(correct_medium[feature], alpha=0.7, label='Medium (correct)', bins=20, color='blue')
                    
                    ax.set_title(f'{feature} Distribution')
                    ax.legend()
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        
        recommendations = []
        
        # Check misclassification rate
        if analysis['easy_to_medium_rate'] > 0.3:  # More than 30% misclassification
            recommendations.append("HIGH PRIORITY: Easy→Medium misclassification rate is high (>30%)")
        
        # Check feature patterns
        problematic_features = []
        for feature, stats in analysis['feature_analysis'].items():
            if stats['easy_misclass_closer_to_medium']:
                problematic_features.append(feature)
        
        if problematic_features:
            recommendations.append(f"Features causing confusion: {', '.join(problematic_features)}")
            recommendations.append("Consider feature engineering or rebalancing training data")
        
        # General recommendations
        recommendations.extend([
            "1. Check class balance in training data - Easy class might be underrepresented",
            "2. Consider adjusting class weights during training to penalize Easy→Medium errors more",
            "3. Review annotation quality - some 'Easy' samples might be mislabeled",
            "4. Add more diverse Easy samples to training data",
            "5. Consider ensemble methods or threshold tuning for Easy class"
        ])
        
        return recommendations

def main():
    """Main analysis function."""
    
    analyzer = PredictionAnalyzer()
    
    # You'll need to create a predictions CSV first
    # This would typically come from running inference on a validation set
    predictions_file = "validation_predictions.csv"
    
    if not Path(predictions_file).exists():
        logger.warning(f"Predictions file {predictions_file} not found.")
        logger.info("To use this analyzer:")
        logger.info("1. Run inference on your validation set")
        logger.info("2. Save results as CSV with columns: filename, true_label, predicted_label, confidence, [feature_scores]")
        logger.info("3. Run this script again")
        return
    
    # Run analysis
    logger.info("Analyzing feature patterns...")
    feature_analysis = analyzer.analyze_feature_patterns(predictions_file)
    
    logger.info("Analyzing confidence patterns...")
    confidence_analysis = analyzer.analyze_confidence_patterns(predictions_file)
    
    logger.info("Creating visualizations...")
    analyzer.plot_confusion_analysis(predictions_file)
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations(feature_analysis)
    
    # Save results
    results = {
        'feature_analysis': feature_analysis,
        'confidence_analysis': confidence_analysis,
        'recommendations': recommendations
    }
    
    with open('misclassification_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EASY→MEDIUM MISCLASSIFICATION ANALYSIS")
    print("="*50)
    print(f"Misclassification rate: {feature_analysis['easy_to_medium_rate']:.2%}")
    print(f"Total misclassifications: {feature_analysis['misclassification_count']}")
    
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\nDetailed results saved to: misclassification_analysis.json")
    print(f"Plots saved to: analysis_plots/")

if __name__ == "__main__":
    main()
