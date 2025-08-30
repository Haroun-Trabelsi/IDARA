import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os 
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import sys
import subprocess

# Check and install required packages
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
        return True
    except Exception as e:
        print(f"Failed to install {package}: {e}")
        return False

# Install required packages
required_packages = ["torch", "torchvision", "pandas"]
for package in required_packages:
    try:
        if package == "torch":
            import torch
            print("torch package is already installed")
        elif package == "torchvision":
            import torchvision
            print("torchvision package is already installed")
        elif package == "pandas":
            import pandas
            print("pandas package is already installed")
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

# Model Definition (same as training)
class FocusPullModel(nn.Module):
    def __init__(self, base_model):
        super(FocusPullModel, self).__init__()
        self.base = base_model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, dropout=0.7, batch_first=True)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.base(c_in)
        c_out = self.pool(c_out)
        c_out = c_out.view(batch_size, timesteps, 2048)
        r_out, _ = self.lstm(c_out)
        r_out = r_out[:, -1, :]
        out = self.fc(r_out)
        return out

# Inference Transform (consistent with training)
inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to analyze video for focus pulls
def analyze_focus_pull(video_path, model_path, seq_length=10, threshold=0.5, save_output=False, output_dir=None):
    try:
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model with debug information
        base_model = models.resnet50(pretrained=False)
        base_model = nn.Sequential(*list(base_model.children())[:-1])  # Match training: remove final FC layer
        model = FocusPullModel(base_model).to(device)
        
        # Load and debug state dictionary
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # Allow loading with missing keys if intentional
        model.eval()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return 0.0, False, 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Setup output video if needed
        output_path = None
        out = None
        if save_output and output_dir:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_name = os.path.basename(video_path)
            output_path = os.path.join(output_dir, f"focus_pull_{video_name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames = []
        scores = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = inference_transforms(frame_rgb).to(device)
            frames.append(frame_tensor)

            if len(frames) == seq_length or frame_idx == total_frames - 1:
                if len(frames) < seq_length:
                    # Pad with the last frame if not enough for a full sequence
                    while len(frames) < seq_length:
                        frames.append(frame_tensor)
                
                seq_frames = torch.stack(frames).unsqueeze(0)  # [1, seq_length, 3, 224, 224]
                with torch.no_grad():
                    outputs = model(seq_frames)
                    score = torch.softmax(outputs, dim=1)[0, 1].item()  # Probability of focus pull
                scores.append(score)

                # Write to output video if enabled
                if save_output and out and out.isOpened():
                    # Annotate the current frame
                    label = "Focus Pull" if score > threshold else "No Focus Pull"
                    text = f"Score: {score:.3f} ({label})"
                    color = (0, 0, 255) if score > threshold else (0, 255, 0)
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    out.write(frame)

                frames = frames[1:]  # Slide window (keep the last frame for next sequence)

            elif save_output and out and out.isOpened():
                out.write(frame)

            frame_idx += 1

        cap.release()
        if out:
            out.release()

        avg_score = np.mean(scores) if scores else 0.0
        focus_pull_detected = avg_score > threshold
        return avg_score, focus_pull_detected, total_frames
    
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return 0.0, False, 0

def batch_process_videos(input_path, output_csv, model_path, seq_length=10, threshold=0.5, 
                         save_output=False, output_dir=None, max_workers=4, file_extensions=None):
    """
    Process multiple videos in a directory or a single video file.
    
    Args:
        input_path (str): Path to a video file or directory containing videos
        output_csv (str): Path for output CSV file
        model_path (str): Path to the trained model
        seq_length (int): Length of sequence for LSTM
        threshold (float): Threshold for focus pull detection
        save_output (bool): Whether to save annotated output videos
        output_dir (str): Directory to save output videos
        max_workers (int): Maximum number of parallel workers
        file_extensions (list): List of file extensions to process
        
    Returns:
        pd.DataFrame: DataFrame containing results for all processed videos
    """
    if file_extensions is None:
        file_extensions = ['.mp4', '.mpv', '.avi', '.mov', '.MOV']
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if save_output and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get list of video files to process
    video_files = []
    if os.path.isfile(input_path):
        if any(input_path.lower().endswith(ext) for ext in file_extensions):
            video_files = [input_path]
    else:
        for ext in file_extensions:
            video_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            video_files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
    
    if not video_files:
        print(f"No video files found in {input_path} with extensions {file_extensions}")
        return pd.DataFrame()
    
    print(f"Found {len(video_files)} video files to process")
    
    # Define a function to process a single video
    def process_video(video_path):
        try:
            print(f"Processing: {os.path.basename(video_path)}")
            avg_score, detected, total_frames = analyze_focus_pull(
                video_path, 
                model_path=model_path, 
                seq_length=seq_length,
                threshold=threshold,
                save_output=save_output,
                output_dir=output_dir
            )
            print(f"Processed: {os.path.basename(video_path)} - Focus Pull Score: {avg_score:.4f}")
            return {
                "file_name": video_path,
                "focus_pull_score": avg_score,
                "focus_pull_detected": detected,
                "total_frames": total_frames
            }
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {
                "file_name": video_path,
                "focus_pull_score": 0.0,
                "focus_pull_detected": False,
                "total_frames": 0,
                "error": str(e)
            }
    
    # Process videos in parallel or sequentially
    results = []
    if max_workers > 1 and len(video_files) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(executor.map(process_video, video_files))
            results = futures
    else:
        for video_path in video_files:
            result = process_video(video_path)
            results.append(result)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Create a simplified DataFrame with just filename and focus pull score
    simple_df = pd.DataFrame({
        "file_name": results_df["file_name"],
        "focus_pull_score": results_df["focus_pull_score"]
    })
    
    # Save results to CSV
    simple_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary
    valid_results = results_df.dropna(subset=['focus_pull_score'])
    if not valid_results.empty:
        print("\n--- Summary Statistics ---")
        print(f"Total videos processed: {len(results_df)}")
        print(f"Videos successfully analyzed: {len(valid_results)}")
        print(f"Average focus pull score: {valid_results['focus_pull_score'].mean():.4f}")
        print(f"Videos with detected focus pull: {sum(valid_results['focus_pull_detected'])}")
        
    return results_df

if __name__ == "__main__":
    # Set your paths directly in the code
    input_path = "D:/IDARA TEST/V02 - DATA/EASY"  # Path to folder containing videos
    output_csv = "D:\IDARA TEST/focus_pull_scores_nouveau_dossier.csv"  # Path for output CSV file
    model_path = r"D:\Projects_MP4\Clients\Parallaxe plates\segment-anything\focus_pull_resnet50_lstm.pth"  # Path to the trained model - using raw string to avoid control character issues
    
    # Parameters
    seq_length = 10
    threshold = 0.5
    save_output = False  # Set to True if you want to save annotated videos
    output_dir = "D:\IDARA TEST/focus_pull_outputs" if save_output else None
    max_workers = 1  # Reduced to 1 to avoid path issues with multiple threads
    file_extensions = ['.mp4', '.mpv', '.avi', '.mov', '.MOV']
    
    print(f"Input path: {input_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Model path: {model_path}")
    
    # Check if model file exists before processing
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please check the path and make sure the file exists.")
        sys.exit(1)
    else:
        print(f"Model file found: {model_path}")
    
    results = batch_process_videos(
        input_path=input_path,
        output_csv=output_csv,
        model_path=model_path,
        seq_length=seq_length,
        threshold=threshold,
        save_output=save_output,
        output_dir=output_dir,
        max_workers=max_workers,
        file_extensions=file_extensions
    )