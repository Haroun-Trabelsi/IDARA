import torch
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer
from joblib import load
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional

class ComprehensiveModelEvaluator:
    def __init__(self, model_path: str, scaler_path: str, imputer_path: str):
        """
        Initialize the evaluator with your pretrained model and preprocessors
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # If the checkpoint is a dict with model weights, rebuild the architecture
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            try:
                from train_model import MultimodalRNN  # deferred import to avoid heavy cost if unused
            except ImportError as e:
                raise RuntimeError("MultimodalRNN class not found. Ensure train_model.py is in PYTHONPATH") from e

            cfg = checkpoint.get('config', {})
            self.model = MultimodalRNN(
                vis_feat_dim=cfg.get('vis_feat_dim', 2048),
                rnn_hidden_dim=cfg.get('rnn_hidden_dim', 256),
                dense_hidden_dim=cfg.get('dense_hidden_dim', 128),
                num_classes=cfg.get('num_classes', 3),
                rnn_type=cfg.get('rnn_type', 'lstm')
            )
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # The checkpoint is the full model object
            self.model = checkpoint

        # Move to device & set eval mode
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessors
        self.scaler = load(scaler_path)
        self.imputer = load(imputer_path)
        
    def preprocess_data(self, X):
        """Apply the same preprocessing as during training"""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled
    
    def get_predictions(self, X):
        """Get model predictions"""
        X_processed = self.preprocess_data(X)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        # Handle different output formats
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
            
        return predictions
    
    def evaluate_regression(self, X_test, y_test) -> Dict[str, float]:
        """Comprehensive regression metrics"""
        y_pred = self.get_predictions(X_test)
        
        # Handle multi-output case
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = y_pred.flatten()
            y_test = y_test.flatten()
        else:
            y_pred = y_pred.ravel()
            y_test = y_test.ravel()
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
        
        # Additional metrics
        metrics['max_error'] = max_error(y_test, y_pred)
        metrics['mean_squared_log_error'] = mean_squared_log_error(
            np.abs(y_test), np.abs(y_pred)) if np.all(y_test >= 0) and np.all(y_pred >= 0) else np.nan
        
        # Percentage-based metrics
        metrics['mape'] = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        metrics['smape'] = np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + 1e-8)) * 100
        
        # Statistical metrics
        residuals = y_test - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['skewness_residual'] = stats.skew(residuals)
        metrics['kurtosis_residual'] = stats.kurtosis(residuals)
        
        # Correlation metrics
        metrics['pearson_corr'] = stats.pearsonr(y_test, y_pred)[0]
        metrics['spearman_corr'] = stats.spearmanr(y_test, y_pred)[0]
        
        return metrics
    
    def evaluate_classification(self, X_test, y_test) -> Dict[str, Any]:
        """Comprehensive classification metrics"""
        y_pred_proba = self.get_predictions(X_test)
        
        # Handle different prediction formats
        if len(y_pred_proba.shape) > 1:
            if y_pred_proba.shape[1] == 1:  # Binary classification with single output
                y_pred_proba = torch.sigmoid(torch.FloatTensor(y_pred_proba)).numpy()
                y_pred = (y_pred_proba > 0.5).astype(int).ravel()
            else:  # Multi-class
                y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_test, y_pred, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_test, y_pred, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_test, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Additional metrics
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_test, y_pred)
        
        # Probability-based metrics (if applicable)
        if len(y_pred_proba.shape) > 1 or np.max(y_pred_proba) <= 1:
            try:
                if len(np.unique(y_test)) == 2:  # Binary classification
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                        y_pred_proba_binary = y_pred_proba[:, 1]
                    else:
                        y_pred_proba_binary = y_pred_proba.ravel()
                    
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba_binary)
                    metrics['average_precision'] = average_precision_score(y_test, y_pred_proba_binary)
                    metrics['log_loss'] = log_loss(y_test, y_pred_proba_binary)
                else:  # Multi-class
                    if len(y_pred_proba.shape) > 1:
                        metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, 
                                                              multi_class='ovr', average='macro')
                        metrics['log_loss'] = log_loss(y_test, y_pred_proba)
            except Exception as e:
                print(f"Could not compute probability-based metrics: {e}")
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        return metrics
    
    def evaluate_model(self, X_test, y_test, task_type: str = 'auto') -> Dict[str, Any]:
        """
        Main evaluation function
        Args:
            X_test: Test features
            y_test: Test targets
            task_type: 'regression', 'classification', or 'auto'
        """
        if task_type == 'auto':
            # Try to infer task type
            unique_values = len(np.unique(y_test))
            if unique_values <= 20 and np.all(y_test == y_test.astype(int)):
                task_type = 'classification'
                print(f"Auto-detected task type: classification ({unique_values} classes)")
            else:
                task_type = 'regression'
                print(f"Auto-detected task type: regression")
        
        print(f"Evaluating model for {task_type} task...")
        
        if task_type == 'regression':
            return self.evaluate_regression(X_test, y_test)
        elif task_type == 'classification':
            return self.evaluate_classification(X_test, y_test)
        else:
            raise ValueError("task_type must be 'regression', 'classification', or 'auto'")
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for key, value in metrics.items():
            if key == 'confusion_matrix':
                print(f"\n{key.upper()}:")
                print(value)
            elif isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    
    def create_evaluation_report(self, X_test, y_test, task_type='auto', 
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report
        """
        metrics = self.evaluate_model(X_test, y_test, task_type)
        
        # Create visualizations for regression
        if task_type == 'regression' or (task_type == 'auto' and 
                                        len(np.unique(y_test)) > 20):
            self._create_regression_plots(X_test, y_test)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Save report if requested
        if save_path:
            report_df = pd.DataFrame(list(metrics.items()), 
                                   columns=['Metric', 'Value'])
            report_df.to_csv(save_path, index=False)
            print(f"\nReport saved to: {save_path}")
        
        return metrics
    
    def _create_regression_plots(self, X_test, y_test):
        """Create regression diagnostic plots"""
        y_pred = self.get_predictions(X_test).ravel()
        y_test = y_test.ravel()
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.show()

# Data preparation utilities
def prepare_test_data_from_full_dataset(data_path: str, target_column: str, 
                                      test_size: float = 0.2, random_state: int = 42):
    """
    Split your full dataset into train/test if you only have one dataset
    """
    from sklearn.model_selection import train_test_split
    
    # Load your full dataset
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y 
        if len(np.unique(y)) <= 10 else None  # Stratify for classification
    )
    
    print(f"Created test set with {len(X_test)} samples ({test_size*100}% of data)")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def prepare_test_data_from_separate_files(features_path: str, targets_path: str):
    """
    Load test data from separate feature and target files
    """
    X_test = pd.read_csv(features_path)
    y_test = pd.read_csv(targets_path)
    
    # Handle different target file formats
    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]  # Convert to series if single column
    
    print(f"Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    return X_test, y_test

def create_holdout_set(data_path: str, target_column: str, holdout_size: int = 100):
    """
    Create a small holdout set for quick evaluation
    """
    df = pd.read_csv(data_path)
    
    # Take a random sample
    holdout_df = df.sample(n=min(holdout_size, len(df)), random_state=42)
    
    X_test = holdout_df.drop(columns=[target_column])
    y_test = holdout_df[target_column]
    
    print(f"Created holdout set with {len(X_test)} samples")
    return X_test, y_test

def prepare_validation_data(data_path: str, target_column: str, 
                          train_size: float = 0.6, val_size: float = 0.2, 
                          test_size: float = 0.2, random_state: int = 42):
    """
    Create train/validation/test split
    """
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Usage examples
def main():
    # Initialize evaluator
    evaluator = ComprehensiveModelEvaluator(
        model_path="inference_model/best_vfx_model.pt",
        scaler_path="inference_model/static_scaler.pkl",
        imputer_path="inference_model/static_imputer.pkl"
    )
    
    print("Choose one of the following data preparation methods:")
    print("\n1. SPLIT FROM FULL DATASET (most common)")
    print("   Use this if you have one dataset file and need to create test data")
    print("   X_test, y_test = prepare_test_data_from_full_dataset(")
    print("       'your_dataset.csv', 'target_column_name', test_size=0.2)")
    
    print("\n2. SEPARATE FILES")
    print("   Use this if you already have separate feature and target files")
    print("   X_test, y_test = prepare_test_data_from_separate_files(")
    print("       'test_features.csv', 'test_targets.csv')")
    
    print("\n3. QUICK HOLDOUT")
    print("   Use this for quick evaluation with a small sample")
    print("   X_test, y_test = create_holdout_set(")
    print("       'your_dataset.csv', 'target_column_name', holdout_size=100)")
    
    print("\n4. TRAIN/VAL/TEST SPLIT")
    print("   Use this to create proper train/validation/test splits")
    print("   X_train, X_val, X_test, y_train, y_val, y_test = prepare_validation_data(")
    print("       'your_dataset.csv', 'target_column_name')")
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE:")
    print("="*60)
    print("# Replace 'your_dataset.csv' and 'target_column' with your actual values")
    print("X_test, y_test = prepare_test_data_from_full_dataset(")
    print("    'your_dataset.csv', 'target_column', test_size=0.2)")
    print("")
    print("# Then evaluate your model")
    print("metrics = evaluator.create_evaluation_report(")
    print("    X_test, y_test, task_type='auto', save_path='evaluation_report.csv')")
    
    return evaluator

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Benchmark MultimodalRNN and baselines")
    parser.add_argument("--features", required=True, help="Path to CSV containing test features")
    parser.add_argument("--labels", required=True, help="Path to CSV containing test labels (single column)")
    parser.add_argument("--task", choices=["auto", "classification", "regression"], default="auto",
                        help="Task type; auto will try to infer from labels")
    parser.add_argument("--out", default="evaluation_report.csv", help="Where to write the CSV report")
    parser.add_argument("--model", default="inference_model/best_vfx_model.pt", help="Path to model checkpoint")
    parser.add_argument("--scaler", default="inference_model/static_scaler.pkl", help="Path to fitted scaler (.pkl)")
    parser.add_argument("--imputer", default="inference_model/static_imputer.pkl", help="Path to fitted imputer (.pkl)")

    args = parser.parse_args()

    # Load data
    X_test = pd.read_csv(args.features)
    y_test_df = pd.read_csv(args.labels)
    if y_test_df.shape[1] == 1:
        y_test = y_test_df.iloc[:, 0]
    else:
        raise ValueError("Labels CSV must have exactly one column with targets")

    # Instantiate evaluator
    evaluator = ComprehensiveModelEvaluator(args.model, args.scaler, args.imputer)

    # Run evaluation
    report = evaluator.create_evaluation_report(X_test, y_test, task_type=args.task, save_path=args.out)

    print("\n✔ Evaluation complete. Report saved to", os.path.abspath(args.out))