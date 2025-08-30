#!/usr/bin/env python3
"""
Debug script to understand why Easy detection isn't working.
This will help us see the actual feature values and rule evaluation.
"""

import numpy as np
import yaml
from pathlib import Path
import json
from typing import Dict, Any

def load_config():
    """Load the config to get feature names."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def debug_easy_detection(complexity_scores: Dict[str, float], sequence_features: np.ndarray = None):
    """Debug the Easy detection logic with actual values."""
    
    config = load_config()
    feature_names = config['multimodal_model']['static_features']
    
    print("=" * 60)
    print("EASY DETECTION DEBUG")
    print("=" * 60)
    
    # Simulate static feature preparation (like in ModelPredictor)
    static_data = [complexity_scores.get(name, 0.0) for name in feature_names[:-1]]
    
    # Add sequence feature if provided
    if sequence_features is not None:
        sequence_feature = np.mean(sequence_features)
        static_data.append(sequence_feature)
    else:
        static_data.append(0.0)  # Default
    
    static_features = np.array(static_data).reshape(1, -1).astype(np.float32)
    
    print(f"Feature names: {feature_names}")
    print(f"Static features: {static_features[0].tolist()}")
    print()
    
    # Create feature dict
    features = {name: static_features[0][i] for i, name in enumerate(feature_names)}
    
    print("INDIVIDUAL FEATURE VALUES:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    print()
    
    # Define Easy conditions (same as in code)
    easy_conditions = {
        'zoom_score': 0.4,
        'blur_score': 0.4,
        'motion_score': 0.4,
        'noise_score': 0.4,
        'distortion_score': 0.4,
        'overlap_score': 0.5,
        'light_score': 0.5
    }
    
    print("EASY CONDITION EVALUATION:")
    conditions_met = []
    for feature, threshold in easy_conditions.items():
        if feature in features:
            value = features[feature]
            is_low = value < threshold
            conditions_met.append(is_low)
            status = "✓ PASS" if is_low else "✗ FAIL"
            print(f"  {feature}: {value:.4f} < {threshold} → {status}")
        else:
            print(f"  {feature}: NOT FOUND")
    
    easy_conditions_count = sum(conditions_met)
    total_conditions = len(conditions_met)
    
    print(f"\nCONDITIONS SUMMARY:")
    print(f"  Conditions met: {easy_conditions_count}/{total_conditions}")
    print(f"  Percentage: {easy_conditions_count/total_conditions*100:.1f}%")
    
    # Test different override scenarios
    print(f"\nOVERRIDE SCENARIOS:")
    
    # Test with different model probabilities
    test_model_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    for model_easy_prob in test_model_probs:
        strong_easy_signal = (easy_conditions_count >= 4 and model_easy_prob < 0.3)
        moderate_easy_signal = (easy_conditions_count >= 6 and model_easy_prob < 0.5)
        overwhelming_easy_signal = (easy_conditions_count >= 7)
        
        would_override = (total_conditions >= 5 and 
                         (strong_easy_signal or moderate_easy_signal or overwhelming_easy_signal))
        
        override_reason = "none"
        if overwhelming_easy_signal:
            override_reason = "overwhelming"
        elif moderate_easy_signal:
            override_reason = "moderate"
        elif strong_easy_signal:
            override_reason = "strong"
        
        status = "YES" if would_override else "NO"
        print(f"  Model Easy={model_easy_prob:.1f} → Override: {status} ({override_reason})")
    
    print("\n" + "=" * 60)
    
    return {
        'features': features,
        'conditions_met': easy_conditions_count,
        'total_conditions': total_conditions,
        'conditions_percentage': easy_conditions_count/total_conditions*100 if total_conditions > 0 else 0
    }

def main():
    """Main debug function."""
    
    print("EASY DETECTION DEBUGGER")
    print("This will help diagnose why Easy sequences aren't being detected.")
    print()
    
    # Example 1: Manually input scores for testing
    print("EXAMPLE 1: Manual test scores (all low values)")
    test_scores_low = {
        'zoom_score': 0.1,
        'blur_score': 0.1,
        'motion_score': 0.1,
        'noise_score': 0.1,
        'distortion_score': 0.1,
        'overlap_score': 0.1,
        'light_score': 0.1,
        'parallax_score': 0.1,
        'focus_pull_score': 0.1
    }
    
    result1 = debug_easy_detection(test_scores_low)
    
    print("\nEXAMPLE 2: Manual test scores (medium values)")
    test_scores_medium = {
        'zoom_score': 0.5,
        'blur_score': 0.5,
        'motion_score': 0.5,
        'noise_score': 0.5,
        'distortion_score': 0.5,
        'overlap_score': 0.6,
        'light_score': 0.6,
        'parallax_score': 0.5,
        'focus_pull_score': 0.5
    }
    
    result2 = debug_easy_detection(test_scores_medium)
    
    print("\nTO DEBUG YOUR ACTUAL SEQUENCE:")
    print("1. Check your logs for 'Static features (raw):' to see actual values")
    print("2. Copy those values and run:")
    print("   debug_easy_detection({'zoom_score': X, 'blur_score': Y, ...})")
    print("3. Look for which conditions are failing")
    
    print("\nCOMMON ISSUES:")
    print("- Features might be scaled/normalized differently than expected")
    print("- Some complexity scores might be consistently high")
    print("- The sequence_mean feature might be throwing off the detection")
    print("- Model preprocessing (scaler/imputer) might be changing values")

if __name__ == "__main__":
    main()
