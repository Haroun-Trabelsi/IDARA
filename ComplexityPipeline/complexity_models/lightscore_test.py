import os
import cv2
import numpy as np
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

def analyze_light_score(video_path):
    """
    Analyze the light score of a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary containing light score analysis results
    """
    try:
        print(f"Analyzing Light Score for: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open {video_path}")
            return {
                "error": f"Cannot open {video_path}",
                "avg_light_score": 0.0,
                "min_light_score": 0.0,
                "max_light_score": 0.0
            }
        
        scores = []
        histograms = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames to speed up processing for large videos
        sample_rate = max(1, total_frames // 100)  # Process ~100 frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process only every nth frame
            if frame_count % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Normalized light score between 0 and 1
                score = np.mean(gray) / 255.0
                scores.append(score)
                # Histogram: 256 bins for grayscale
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
                histograms.append(hist)
                
            frame_count += 1
            
        cap.release()
        
        if not scores:
            return {
                "error": "No frames processed.",
                "avg_light_score": 0.0,
                "min_light_score": 0.0,
                "max_light_score": 0.0
            }
            
        # Compute average histogram
        avg_histogram = np.mean(histograms, axis=0) if histograms else np.zeros(256)
        
        return {
            'light_score': scores,  # Now normalized 0..1
            'avg_light_score': float(np.mean(scores)),
            'min_light_score': float(np.min(scores)),
            'max_light_score': float(np.max(scores)),
            'avg_histogram': avg_histogram.tolist()
        }
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return {
            "error": f"Analysis failed: {e}",
            "avg_light_score": 0.0,
            "min_light_score": 0.0,
            "max_light_score": 0.0
        }

def batch_process_videos(input_path, output_csv, max_workers=4, file_extensions=None):
    """
    Process multiple videos in a directory or a single video file.
    
    Args:
        input_path (str): Path to a video file or directory containing videos
        output_csv (str): Path for output CSV file
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
            result = analyze_light_score(video_path)
            print(f"Processed: {os.path.basename(video_path)} - Light Score: {result['avg_light_score']:.4f}")
            return {
                "file_name": video_path,
                "light_score": result['avg_light_score'],
                "min_light_score": result['min_light_score'],
                "max_light_score": result['max_light_score']
            }
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {
                "file_name": video_path,
                "light_score": 0.0,
                "min_light_score": 0.0,
                "max_light_score": 0.0,
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
    
    # Create a simplified DataFrame with just filename and light score
    simple_df = pd.DataFrame({
        "file_name": results_df["file_name"],
        "light_score": results_df["light_score"]
    })
    
    # Save results to CSV
    simple_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary
    valid_results = results_df.dropna(subset=['light_score'])
    if not valid_results.empty:
        print("\n--- Summary Statistics ---")
        print(f"Total videos processed: {len(results_df)}")
        print(f"Videos successfully analyzed: {len(valid_results)}")
        print(f"Average light score: {valid_results['light_score'].mean():.4f}")
        print(f"Min light score: {valid_results['light_score'].min():.4f}")
        print(f"Max light score: {valid_results['light_score'].max():.4f}")
        
    return results_df

if __name__ == "__main__":
    # Set your paths directly in the code
    input_path = "D:\IDARA TEST/Nouveau dossier"  # Path to folder containing videos
    output_csv = "D:\IDARA TEST/light_scores_Nouveau_dossier.csv"  # Path for output CSV file
    
    # Parameters
    max_workers = 4
    file_extensions = ['.mp4', '.mpv', '.avi', '.mov', '.MOV']
    
    print(f"Input path: {input_path}")
    print(f"Output CSV: {output_csv}")
    
    results = batch_process_videos(
        input_path=input_path,
        output_csv=output_csv,
        max_workers=max_workers,
        file_extensions=file_extensions
    )
