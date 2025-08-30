# predict_vfx.py
from estimation import VFXBidPredictor
from estimation import VFXTemplateAnalyzer
import os

# Initialize predictor
predictor = VFXBidPredictor()

# Load your trained model (update path)
model_path = "vfx_bid_predictor_your_data.joblib"  # <<< CHANGE TO YOUR .joblib PATH
predictor.load_model(model_path)

# Sample input (adapt categories to match your training data)
sample_input = {
    "description": "4K RED footage, lens grid provided, 3D tracking needed",
    "complexity_task": "Medium",       # Must match trained categories
    "task_name": "CamTrack",        # Must match trained categories
    "project_name": "7760",  # Must match trained categories
    "notes_count": 5
}

# Predict
predicted_hours = predictor.predict_new_descriptions(
    [sample_input["description"]],
    {
        "complexity_task": [sample_input["complexity_task"]],
        "task_name": [sample_input["task_name"]],
        "project_name": [sample_input["project_name"]],
        "notes_count": [sample_input["notes_count"]]
    }
)

print(f"Predicted VFX hours: {predicted_hours[0]:.1f}")