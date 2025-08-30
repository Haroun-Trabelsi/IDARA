import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision # Import torchvision directly to access its __version__ attribute
from torchvision import transforms, models 
import os
import csv
import sys

class NoiseClassificationCNNLSTM(nn.Module):
    def __init__(self, num_features=1280, hidden_size=256, num_layers=2):
        super(NoiseClassificationCNNLSTM, self).__init__()
        # Load MobileNetV2 without pre-trained weights for custom training
        # If your model was trained with ImageNet weights, set weights=models.MobileNetV2_Weights.IMAGENET1K_V1
        self.cnn = models.mobilenet_v2(weights=None) 
        # Remove the classifier (last layer) to get features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling to get a fixed size feature vector

        self.fc_flow = nn.Linear(3, 32) # Maps 3 optical flow stats to a 32-dim feature vector

        # LSTM takes combined features (CNN features + Flow features)
        # num_features is the output size of MobileNetV2's feature extractor (1280 for the default output)
        self.lstm = nn.LSTM(num_features + 32, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Output a single score
        self.sigmoid = nn.Sigmoid() # Sigmoid for binary classification probability

    def forward(self, frames, flow_stats):
        # frames: (batch_size, seq_len, C, H, W)
        # flow_stats: (batch_size, seq_len, 3)

        batch_size, seq_len, C, H, W = frames.size()
        
        # Process each frame through the CNN
        cnn_features = []
        for t in range(seq_len):
            frame = frames[:, t] # Get the t-th frame for all batches
            feat = self.cnn(frame) # Pass through CNN backbone
            feat = self.pool(feat) # Apply global average pooling
            feat = feat.view(batch_size, -1) # Flatten to (batch_size, num_features)
            cnn_features.append(feat)
        cnn_features = torch.stack(cnn_features, dim=1) # Stack into (batch_size, seq_len, num_features)

        # Process flow stats through the linear layer
        # flow_stats: (batch_size, seq_len, 3) -> fc_flow -> (batch_size, seq_len, 32)
        flow_features = self.fc_flow(flow_stats)

        # CORRECTED LINE: Concatenate CNN features and flow features directly
        # Both cnn_features and flow_features are already (batch_size, seq_len, D)
        combined = torch.cat([cnn_features, flow_features], dim=2) 
        
        # Pass combined features through LSTM
        lstm_out, _ = self.lstm(combined) 
        
        # Take the output from the last time step for classification
        out = self.fc(lstm_out[:, -1, :]) 
        
        return self.sigmoid(out)

def compute_optical_flow(prev_frame, curr_frame):
    """Compute optical flow between two frames and return flow statistics"""
    try:
        # Convert to grayscale and resize to a consistent size for flow calculation
        # (224, 224) is a common choice for input to CNNs, so keeping it consistent helps.
        prev_frame_gray = cv2.cvtColor(cv2.resize(prev_frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        curr_frame_gray = cv2.cvtColor(cv2.resize(curr_frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow using Farneback method
        # Parameters chosen for typical motion tracking; adjust if needed for specific noise types.
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert to polar coordinates: magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Return statistics: mean magnitude, std dev magnitude, mean angle
        # These provide a compact representation of motion
        return np.array([np.mean(mag), np.std(mag), np.mean(ang)])
    except Exception as e:
        # Added robust error handling for flow computation
        print(f"  WARNING: Error computing optical flow: {e}. Returning default values [0.0, 0.0, 0.0].") 
        return np.array([0.0, 0.0, 0.0]) 

def analyze_video_noise(video_path, output_path=None, model_path="noise_classification_cnn_lstm.pt", 
                        seq_length=5, threshold=0.5, sample_rate=1, save_output=False):
    """
    Analyze video noise and return noise score.
    If save_output is True, creates an annotated video file.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str, optional): Path to save the annotated output video. Required if save_output is True.
        model_path (str, optional): Path to the trained PyTorch model.
        seq_length (int, optional): Number of frames in a sequence for the LSTM. Defaults to 5.
        threshold (float, optional): Noise score threshold for classification (Noisy/Noise-free). Defaults to 0.5.
        sample_rate (int, optional): Process every Nth frame. Defaults to 1 (process all frames).
        save_output (bool, optional): Whether to save an annotated video. Defaults to False.

    Returns:
        tuple: (avg_noise_score, noise_detected_flag, total_frames_processed, output_video_path)
    """
    print(f"Analyzing Noise for: {os.path.basename(video_path)}")
    
    # --- DEBUGGING START: Explicit path and file checks ---
    print(f"  DEBUG: Checking video file existence: {os.path.abspath(video_path)}")
    if not os.path.exists(video_path):
        print(f"  ERROR: Video file NOT FOUND: {video_path}")
        return 0.0, False, 0, None
    print(f"  DEBUG: Video file found.")

    print(f"  DEBUG: Checking model file existence: {os.path.abspath(model_path)}")
    if not os.path.exists(model_path):
        print(f"  ERROR: Model file NOT FOUND: {model_path}")
        return 0.0, False, 0, None
    print(f"  DEBUG: Model file found.")
    # --- DEBUGGING END ---

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = NoiseClassificationCNNLSTM().to(device)
    
    try:
        # Load the model's state_dict
        model_data = torch.load(model_path, map_location=device)
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            # If the checkpoint is a dict containing 'state_dict'
            model.load_state_dict(model_data['state_dict'])
        else:
            # Assume it's a raw state_dict
            model.load_state_dict(model_data)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"  ERROR: Error loading model from {model_path}: {e}")
        print("  WARNING: Using uninitialized model for inference! Scores will be meaningless.")
        return 0.0, False, 0, None
    
    model.eval() # Set model to evaluation mode

    # Define transforms (match training script's normalization)
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts HWC (0-255) to CHW (0.0-1.0)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])

    # Open input video
    print(f"  DEBUG: Attempting to open video with OpenCV: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  CRITICAL ERROR: Could not open video with OpenCV: {video_path}")
        print("  This often means the video file is corrupted, not supported by your OpenCV build (missing codecs like FFMPEG), or path issues.")
        return 0.0, False, 0, None
    print(f"  DEBUG: Video opened successfully with OpenCV.")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    if total_frames == 0:
        print(f"  WARNING: Video reports 0 frames (could be empty or corrupted). Returning 0.0 score.")
        cap.release()
        return 0.0, False, 0, None


    # Initialize output video writer if save_output is True
    out = None
    if save_output and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Create output directory if it doesn't exist
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not open output video writer for {output_path}")
            cap.release()
            return 0.0, False, 0, None

    # Buffers to store frames and optical flow stats for sequence processing
    frame_buffer = [] # Stores RGB frames (resized 224x224)
    flow_stat_buffer = [] # Stores optical flow stats

    noise_scores = [] # Stores predicted noise scores
    annotated_frames = [] # Stores frames for output video (if save_output)
    
    frame_idx = 0 # Overall frame counter (from video start)
    processed_inference_frames = 0 # Counter for frames where actual inference was run

    # Initialize previous frame for flow calculation
    prev_frame_for_flow = None 

    while True:
        ret, frame = cap.read() # Read a frame
        if not ret: # End of video or read error
            break # Exit loop if no more frames

        orig_frame = frame.copy() # Keep a copy of the original frame for annotation/output

        # Process frames based on sample_rate
        if frame_idx % sample_rate == 0:
            try:
                # Convert BGR (OpenCV default) to RGB and resize for model input
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                
                # Compute optical flow
                if prev_frame_for_flow is not None:
                    flow_stat = compute_optical_flow(prev_frame_for_flow, frame_resized)
                    flow_stat_buffer.append(flow_stat)
                else:
                    # For the very first frame in the video, there's no previous frame.
                    # Append a dummy flow stat (e.g., zero motion). Crucial for sequence length.
                    flow_stat_buffer.append(np.array([0.0, 0.0, 0.0])) 

                frame_buffer.append(frame_resized) # Add current frame to buffer
                
                # Maintain sequence length: remove oldest if buffer exceeds seq_length
                if len(frame_buffer) > seq_length:
                    frame_buffer.pop(0)
                    flow_stat_buffer.pop(0) 

                # If we have enough frames for a full sequence, perform inference
                if len(frame_buffer) == seq_length:
                    # Prepare sequence frames for model input
                    seq_frames_tensor = torch.stack([transform(f) for f in frame_buffer], dim=0).unsqueeze(0).to(device) # (1, seq_len, 3, 224, 224)
                    
                    # Prepare sequence flow stats for model input
                    # Note: flow_stat_buffer should now always be of length seq_length
                    seq_flow_tensor = torch.from_numpy(np.array(flow_stat_buffer)).float().unsqueeze(0).to(device) # (1, seq_len, 3)

                    # Model Inference
                    with torch.no_grad():
                        try:
                            score = model(seq_frames_tensor, seq_flow_tensor).item()
                            # print(f"  Frame {frame_idx}: Noise score: {score:.4f}") # Uncomment for per-frame score
                            noise_scores.append(score)
                            processed_inference_frames += 1
                        except Exception as e:
                            print(f"  WARNING: Error during model inference at frame {frame_idx}: {e}")
                            # Fallback: if inference fails, use 0.0 or carry over previous score.
                            # This is only a fallback and indicates a problem with the model or input data format.
                            noise_scores.append(0.0) 
                            if len(noise_scores) > 1:
                                noise_scores[-1] = noise_scores[-2] # Carry over previous score
                            processed_inference_frames += 1

                # Update prev_frame_for_flow for the next iteration (for the *next* sampled frame)
                prev_frame_for_flow = frame_resized
            
            except Exception as e:
                print(f"  WARNING: General error processing sampled frame {frame_idx}: {e}")

        # If saving output, annotate the original frame and add to buffer
        if save_output:
            # Only annotate with a valid score if a sequence was processed and a score was generated
            current_score = noise_scores[-1] if noise_scores and len(frame_buffer) == seq_length else 0.0 
            label = "Noisy" if current_score > threshold else "Noise-free"
            text = f"Noise Score: {current_score:.3f} ({label})"
            cv2.putText(orig_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0) if current_score <= threshold else (0, 0, 255), 2, cv2.LINE_AA)
            annotated_frames.append(orig_frame)
        
        frame_idx += 1
        # Print progress
        if frame_idx % max(1, total_frames // 10) == 0:
            print(f"  Progress: {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")

    # Release video capture and writer
    cap.release()
    if out is not None:
        # Ensure all frames are written (especially important if sample_rate > 1)
        for frame in annotated_frames:
            out.write(frame)
        out.release()

    # Compute average noise score
    avg_noise = np.mean(noise_scores) if noise_scores else 0.0
    noise_detected = avg_noise > threshold

    print(f"\nAnalysis complete for {os.path.basename(video_path)}")
    print(f"  Total frames in video: {total_frames}")
    print(f"  Frames processed for inference (after sample_rate and seq_length): {processed_inference_frames}")
    print(f"  Average noise score: {avg_noise:.4f}")
    print(f"  Noise detected (avg > {threshold}): {noise_detected}")

    return avg_noise, noise_detected, total_frames, output_path

def analyze_videos(video_folder, output_csv, model_path="noise_classification_cnn_lstm.pt", output_folder=None, save_output=False):
    """
    Analyze all video files in a folder and save noise scores to a CSV file.
    
    Args:
        video_folder (str): Path to folder containing video files.
        output_csv (str): Path to output CSV file.
        model_path (str): Path to the trained noise classification model.
        output_folder (str, optional): Path to folder for saving annotated output videos (if save_output is True).
        save_output (bool): Whether to save annotated output videos.
    """
    # Check if the folder exists
    print(f"\nDEBUG: Checking video folder existence: {os.path.abspath(video_folder)}")
    if not os.path.exists(video_folder):
        print(f"ERROR: Video folder NOT FOUND: {video_folder}")
        return
    print(f"DEBUG: Video folder found.")
    
    # Create output folder if saving output videos
    if save_output and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Create the directory for the CSV file if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for file in os.listdir(video_folder):
        file_path = os.path.join(video_folder, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print(f"No video files found in {video_folder} with extensions: {video_extensions}")
        return
    
    print(f"Found {len(video_files)} video files to analyze")
    
    # Make sure the model path is absolute
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    print(f"Using model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}")
    
    # Create CSV file and write header
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['video_name', 'noise_score', 'noise_detected', 'total_frames'])
        
        # Process each video
        for i, video_path in enumerate(video_files):
            print(f"\n[{i+1}/{len(video_files)}] Processing {os.path.basename(video_path)}...")
            
            # Define output path if saving output
            output_path = None
            if save_output and output_folder:
                video_name = os.path.basename(video_path)
                output_path = os.path.join(output_folder, f"noise_annotated_{video_name}")
            
            try:
                # Analyze video
                noise_score, noise_detected, total_frames, _ = analyze_video_noise(
                    video_path=video_path,
                    output_path=output_path,
                    model_path=model_path,
                    save_output=save_output
                )
                
                # Extract video name from path
                video_name = os.path.basename(video_path)
                
                # Write to CSV
                csvwriter.writerow([video_name, noise_score, noise_detected, total_frames])
                
                print(f"Completed {video_name} - Noise Score: {noise_score:.4f} - Noisy: {noise_detected}")
            except Exception as e:
                print(f"CRITICAL ERROR analyzing {os.path.basename(video_path)}: {e}")
                # Write error to CSV
                video_name = os.path.basename(video_path)
                csvwriter.writerow([video_name, 0.0, False, 0])
    
    print(f"\nAnalysis complete. Results saved to {output_csv}")

if __name__ == "__main__":
    # Get command line arguments or use defaults
    # Example usage: python your_script_name.py "C:/path/to/videos" "C:/path/to/results.csv" "C:/path/to/model.pt"
    
    # Default paths (adjust these for your environment)
    DEFAULT_VIDEO_FOLDER = "D:\IDARA TEST\RIO SHOTS" 
    DEFAULT_OUTPUT_CSV = "D:/IDARA TEST/noise_scores_RIO_shots.csv"
    DEFAULT_MODEL_PATH = "D:\Projects_MP4\Clients\Parallaxe plates\segment-anything/noise_classification_cnn_lstm.pt"
    
    video_folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_FOLDER
    output_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_CSV
    model_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL_PATH
    
    # Set to True if you want to save annotated output videos
    save_output = False # Set to True to enable saving annotated videos
    output_folder = "output_videos"  # Subfolder within script's directory for annotated videos
    
    # Print system information
    print("=" * 50)
    print("Video Noise Analysis Tool")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}") 
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    
    # Run the analysis
    print(f"Analyzing all videos in folder: {video_folder}")
    print(f"Results will be saved to: {output_csv}")
    print(f"Using model file: {model_path}")
    if save_output:
        print(f"Annotated videos will be saved to: {os.path.abspath(output_folder)}")
    else:
        print("Annotated videos will NOT be saved (save_output is False).")
    
    analyze_videos(
        video_folder=video_folder,
        output_csv=output_csv,
        model_path=model_path,
        output_folder=output_folder,
        save_output=save_output
    )