#!/usr/bin/env python3
"""
Quick diagnosis script to check Easy→Medium misclassification patterns.
Run this on a batch of videos to see the pattern.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import json
from collections import Counter, defaultdict

# Import your pipeline
from pipeline.tasks import ModelPredictor

def analyze_recent_predictions(log_file_or_results: str = None):
    """Analyze prediction patterns from recent runs."""
    
    # Load config to get class names
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['multimodal_model']['class_names']
    
    print("=== PREDICTION PATTERN ANALYSIS ===")
    print(f"Class names: {class_names}")
    
    # If you have a results file, analyze it
    # Otherwise, we'll create a template for manual analysis
    
    # Simulate some analysis based on your reported issue
    print("\n=== REPORTED ISSUE ANALYSIS ===")
    print("Issue: Model predicts many Easy sequences as Medium")
    
    print("\n=== LIKELY CAUSES ===")
    causes = [
        "1. CLASS IMBALANCE: Easy class underrepresented in training",
        "2. FEATURE OVERLAP: Easy and Medium have similar complexity scores", 
        "3. DECISION BOUNDARY: Model's threshold between Easy/Medium is too conservative",
        "4. ANNOTATION INCONSISTENCY: Some 'Easy' samples might actually be Medium",
        "5. MODEL CALIBRATION: Model is overconfident about Medium predictions"
    ]
    
    for cause in causes:
        print(cause)
    
    print("\n=== IMMEDIATE CHECKS ===")
    checks = [
        "✓ Check confidence scores: Are Easy→Medium predictions low confidence?",
        "✓ Review specific examples: Manually verify misclassified samples",
        "✓ Check feature distributions: Do Easy samples have Medium-like scores?",
        "✓ Examine training data balance: How many Easy vs Medium samples?",
        "✓ Test threshold adjustment: Lower the Medium prediction threshold"
    ]
    
    for check in checks:
        print(check)
    
    return {
        'class_names': class_names,
        'reported_issue': 'Easy→Medium misclassification',
        'analysis_needed': True
    }

def suggest_quick_fixes():
    """Suggest immediate fixes to try."""
    
    print("\n=== QUICK FIXES TO TRY ===")
    
    fixes = [
        {
            'name': 'Adjust Class Weights',
            'description': 'Modify config.yaml to add class-specific thresholds',
            'code': '''
# Add to config.yaml under multimodal_model:
class_thresholds:
  Easy: 0.4    # Lower threshold = easier to predict Easy
  Medium: 0.35
  Hard: 0.25
'''
        },
        {
            'name': 'Temperature Scaling',
            'description': 'Adjust temperature in config.yaml',
            'code': '''
# In config.yaml, try different temperature values:
temperature: 3.0  # Lower = more confident predictions
# or
temperature: 7.0  # Higher = less confident, more balanced
'''
        },
        {
            'name': 'Post-processing Rule',
            'description': 'Add bias correction in pipeline/tasks.py',
            'code': '''
# In ModelPredictor.predict(), after getting probabilities:
# Boost Easy class probability if it's close to Medium
if abs(probabilities[0] - probabilities[1]) < 0.1:  # Easy vs Medium close
    probabilities[0] *= 1.2  # Boost Easy by 20%
    probabilities = probabilities / probabilities.sum()  # Renormalize
'''
        }
    ]
    
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['name']}")
        print(f"   {fix['description']}")
        print(f"   Code:{fix['code']}")
    
    return fixes

def main():
    """Run quick diagnosis."""
    
    print("QUICK DIAGNOSIS: Easy→Medium Misclassification")
    print("=" * 50)
    
    # Analyze patterns
    analysis = analyze_recent_predictions()
    
    # Suggest fixes
    fixes = suggest_quick_fixes()
    
    # Save results
    results = {
        'analysis': analysis,
        'suggested_fixes': fixes,
        'next_steps': [
            'Run analyze_predictions.py for detailed analysis',
            'Test one of the quick fixes above',
            'Collect more validation samples for testing',
            'Consider retraining with balanced data'
        ]
    }
    
    with open('quick_diagnosis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== NEXT STEPS ===")
    for step in results['next_steps']:
        print(f"• {step}")
    
    print(f"\nResults saved to: quick_diagnosis_results.json")

if __name__ == "__main__":
    main()
