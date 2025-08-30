import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
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

class BlurCNNLSTM(nn.Module):
    def __init__(self, num_features=1280, hidden_size=256, num_layers=2):
        super(BlurCNNLSTM, self).__init__()
        self.cnn = models.mobilenet_v2(weights=None)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_flow = nn.Linear(3, 32)
        self.lstm = nn.LSTM(num_features + 32, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, frames, flow_stats):
        batch_size, seq_len, C, H, W = frames.size()
        cnn_features = []
        for t in range(seq_len):
            frame = frames[:, t]
            feat = self.cnn(frame)
            feat = self.pool(feat)
            feat = feat.view(batch_size, -1)
            cnn_features.append(feat)
        cnn_features = torch.stack(cnn_features, dim=1)
        flow_features = self.fc_flow(flow_stats)
        combined = torch.cat([cnn_features, flow_features], dim=2)
        lstm_out, _ = self.lstm(combined)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

def compute_optical_flow(prev_frame, curr_frame):
    # Convert to grayscale and resize
    prev_frame = cv2.resize(prev_frame, (224, 224), interpolation=cv2.INTER_AREA)
    curr_frame = cv2.resize(curr_frame, (224, 224), interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.array([np.mean(mag), np.std(mag), np.mean(ang)])

def analyze_camera_motion_blur(video_path, model_path="blur_cnn_lstm_finetuned.pt", seq_length=5, threshold=0.5):
    try:
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BlurCNNLSTM().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return 0.0, False, 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        flow_stats = []
        blur_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # [224, 224, 3]
            frames.append(frame)

            # Compute optical flow if enough frames
            if len(frames) > 1:
                flow_stats.append(compute_optical_flow(frames[-2], frames[-1]))

            # Process sequence if enough frames
            if len(frames) == seq_length:
                # Prepare input
                seq_frames = np.array(frames)  # [seq_length, 224, 224, 3]
                seq_frames = torch.stack([transform(f) for f in seq_frames], dim=0)  # [seq_length, 3, 224, 224]
                seq_flow = np.zeros((seq_length, 3))  # Pad flow_stats
                seq_flow[:len(flow_stats)] = np.array(flow_stats)
                seq_frames = seq_frames.unsqueeze(0).to(device)  # [1, seq_length, 3, 224, 224]
                seq_flow = torch.from_numpy(seq_flow).float().unsqueeze(0).to(device)  # [1, seq_length, 3]

                # Inference
                with torch.no_grad():
                    score = model(seq_frames, seq_flow).item()
                blur_scores.append(score)

                # Remove oldest frame and flow stat
                frames.pop(0)
                if flow_stats:
                    flow_stats.pop(0)

        cap.release()

        # Compute average blur score
        avg_blur = np.mean(blur_scores) if blur_scores else 0.0
        motion_blur_detected = avg_blur > threshold

        return avg_blur, motion_blur_detected, total_frames
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return 0.0, False, 0

def batch_process_videos(input_path, output_csv, model_path="blur_cnn_lstm_finetuned.pt", 
                         threshold=0.5, max_workers=4, file_extensions=None):
    """
    Process multiple videos in a directory or a single video file.
    
    Args:
        input_path (str): Path to a video file or directory containing videos
        output_csv (str): Path for output CSV file
        model_path (str): Path to the trained model
        threshold (float): Threshold for motion blur detection
        max_workers (int): Maximum number of parallel workers
        file_extensions (list): List of file extensions to process
        
    Returns:
        pd.DataFrame: DataFrame containing results for all processed videos
    """
    if file_extensions is None:
        file_extensions = ['.mp4', '.mpv', '.avi', '.mov', '.MOV']
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
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
            avg_blur, detected, total_frames = analyze_camera_motion_blur(
                video_path, model_path=model_path, threshold=threshold
            )
            print(f"Processed: {os.path.basename(video_path)} - Blur Score: {avg_blur:.4f}")
            return {
                "file_name": video_path,
                "blur_score": avg_blur,
                "blur_detected": detected,
                "total_frames": total_frames
            }
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {
                "file_name": video_path,
                "blur_score": 0.0,
                "blur_detected": False,
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
    
    # Create a simplified DataFrame with just filename and blur score
    simple_df = pd.DataFrame({
        "file_name": results_df["file_name"],
        "blur_score": results_df["blur_score"]
    })
    
    # Save results to CSV
    simple_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary
    valid_results = results_df.dropna(subset=['blur_score'])
    if not valid_results.empty:
        print("\n--- Summary Statistics ---")
        print(f"Total videos processed: {len(results_df)}")
        print(f"Videos successfully analyzed: {len(valid_results)}")
        print(f"Average blur score: {valid_results['blur_score'].mean():.4f}")
        print(f"Videos with detected blur: {sum(valid_results['blur_detected'])}")
        
    return results_df

if __name__ == "__main__":
    # Set your paths directly in the code
    input_path = "D:\IDARA TEST/Nouveau dossier"  # Path to folder containing videos
    output_csv = "D:\IDARA TEST/blur_scores_nouveau_dossier.csv"  # Path for output CSV file
    model_path = "D:\Projects_MP4\Clients\Parallaxe plates\segment-anything/blur_cnn_lstm_finetuned.pt"  # Path to the trained model
    
    # Parameters
    threshold = 0.5
    max_workers = 4
    file_extensions = ['.mp4', '.mpv', '.avi', '.mov', '.MOV']
    
    print(f"Input path: {input_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Model path: {model_path}")
    
    results = batch_process_videos(
        input_path=input_path,
        output_csv=output_csv,
        model_path=model_path,
        threshold=threshold,
        max_workers=max_workers,
        file_extensions=file_extensions
    )