import numpy as np
import torch
import joblib
import pytest

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace

# Import the evaluator from benchmark.py
from benchmark import ComprehensiveModelEvaluator


class DummyClassificationModel(torch.nn.Module):
    """A very small feed-forward net outputting class probabilities."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):  # x shape: (batch, features)
        logits = self.linear(x)
        return torch.nn.functional.softmax(logits, dim=-1)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_evaluator_classification_end_to_end(tmp_path, n_classes):
    """Ensure ComprehensiveModelEvaluator loads model & preprocessors and returns metrics."""

    # -----------------------------
    # 1. Generate synthetic data
    # -----------------------------
    n_samples, n_features = 60, 5
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(low=0, high=n_classes, size=n_samples)

    # ---------------------------------
    # 2. Fit and persist preprocessors
    # ---------------------------------
    imputer = SimpleImputer(strategy="mean").fit(X)
    X_imp = imputer.transform(X)
    scaler = StandardScaler().fit(X_imp)

    imputer_path = tmp_path / "imputer.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)

    # -----------------------------
    # 3. Create & save dummy model
    # -----------------------------
    model = DummyClassificationModel(n_features, n_classes).eval()
    model_path = tmp_path / "dummy_model.pt"
    torch.save(model, model_path)

    # ------------------------------------
    # 4. Instantiate evaluator & evaluate
    # ------------------------------------
    evaluator = ComprehensiveModelEvaluator(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        imputer_path=str(imputer_path),
    )

    metrics = evaluator.evaluate_classification(X, y)

    # -----------------------------
    # 5. Basic sanity assertions
    # -----------------------------
    assert "accuracy" in metrics, "Accuracy metric should be present"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "Accuracy should be in [0, 1] range"

    # Confusion matrix shape should be (n_classes, n_classes)
    conf_mat = metrics.get("confusion_matrix")
    assert conf_mat.shape == (n_classes, n_classes)

    # ROC-AUC present for binary or multi-class
    assert "roc_auc" in metrics or "roc_auc_ovr" in metrics
