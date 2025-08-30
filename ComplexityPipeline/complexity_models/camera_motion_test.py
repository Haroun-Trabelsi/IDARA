import cv2
import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import glob
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
required_packages = ["opencv-python", "numpy", "pandas"]
for package in required_packages:
    try:
        if package == "opencv-python":
            import cv2
            print("opencv-python package is already installed")
        elif package == "numpy":
            import numpy
            print("numpy package is already installed")
        elif package == "pandas":
            import pandas
            print("pandas package is already installed")
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

def analyze_camera_motion(video_path, sample_rate=5, resize_to=(320, 240)):
    """
    Fast camera motion analysis using dense optical flow on sampled, resized frames.
    Args:
        video_path (str): Path to the input video file.
        sample_rate (int): Analyze every Nth frame.
        resize_to (tuple): Resize frames to this size (width, height) for analysis.
    Returns:
        dict: {
            'motion_magnitude': [normalized per-frame values],
            'avg_motion': float,
        }
    """
    try:
        print(f"Analyzing Camera Motion for: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            print(f"Failed to read video: {video_path}")
            return {'error': 'Failed to read video', 'avg_motion': 0.0}

        prev_frame = cv2.resize(prev_frame, resize_to)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_vectors = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            frame = cv2.resize(frame, resize_to)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_vectors.append(np.mean(magnitude))
            prev_gray = gray

        cap.release()

        if motion_vectors:
            motion_array = np.array(motion_vectors)
            min_val = np.min(motion_array)
            max_val = np.max(motion_array)
            if max_val == min_val:
                normalized_motion = np.full_like(motion_array, 0.5)
            else:
                normalized_motion = (motion_array - min_val) / (max_val - min_val)
            avg_motion = float(np.mean(normalized_motion))
            return {
                'motion_magnitude': normalized_motion.tolist(),
                'avg_motion': avg_motion,
            }
        else:
            return {'error': 'No motion data', 'avg_motion': 0.0}
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return {'error': str(e), 'avg_motion': 0.0}

def batch_process_videos(input_path, output_csv, sample_rate=5, resize_to=(320, 240), max_workers=4, file_extensions=None):
    """
    Process multiple videos in a directory or a single video file.
    
    Args:
        input_path (str): Path to a video file or directory containing videos
        output_csv (str): Path for output CSV file
        sample_rate (int): Analyze every Nth frame
        resize_to (tuple): Resize frames to this size (width, height) for analysis
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
            result = analyze_camera_motion(video_path, sample_rate=sample_rate, resize_to=resize_to)
            print(f"Processed: {os.path.basename(video_path)} - Motion Score: {result['avg_motion']:.4f}")
            return {
                "file_name": video_path,
                "motion_score": result['avg_motion']
            }
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {
                "file_name": video_path,
                "motion_score": 0.0,
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
    
    # Create a simplified DataFrame with just filename and motion score
    simple_df = pd.DataFrame({
        "file_name": results_df["file_name"],
        "motion_score": results_df["motion_score"]
    })
    
    # Save results to CSV
    simple_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary
    valid_results = results_df.dropna(subset=['motion_score'])
    if not valid_results.empty:
        print("\n--- Summary Statistics ---")
        print(f"Total videos processed: {len(results_df)}")
        print(f"Videos successfully analyzed: {len(valid_results)}")
        print(f"Average motion score: {valid_results['motion_score'].mean():.4f}")
        print(f"Min motion score: {valid_results['motion_score'].min():.4f}")
        print(f"Max motion score: {valid_results['motion_score'].max():.4f}")
        
    return results_df

if __name__ == "__main__":
    # Set your paths directly in the code
    input_path = "D:\IDARA TEST/Nouveau dossier"  # Path to folder containing videos
    output_csv = "D:\IDARA TEST/motion_scores_Nouveau_dossier.csv"  # Path for output CSV file
    
    # Parameters
    sample_rate = 5  # Analyze every 5th frame
    resize_to = (320, 240)  # Resize frames for faster processing
    max_workers = 4
    file_extensions = ['.mp4', '.mpv', '.avi', '.mov', '.MOV']
    
    print(f"Input path: {input_path}")
    print(f"Output CSV: {output_csv}")
    
    results = batch_process_videos(
        input_path=input_path,
        output_csv=output_csv,
        sample_rate=sample_rate,
        resize_to=resize_to,
        max_workers=max_workers,
        file_extensions=file_extensions
    )
  
        