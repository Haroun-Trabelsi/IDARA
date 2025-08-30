import cv2
import numpy as np
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

def analyze_zoom(video_path, sample_rate=5):
    """
    Detect zoom by analyzing scale changes using OpenCV feature matching.
    Returns average zoom factor (1.0 = no zoom, >1.0 = zoom-in, <1.0 = zoom-out) and zoom_detected boolean.
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return {
                'error': f"Cannot open video: {video_path}",
                'avg_zoom': 1.0,
                'min_zoom': 1.0,
                'max_zoom': 1.0,
                'zoom_detected': False
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, sample_rate)  # Ensure valid sample rate

        scale_factors = []
        frame_indices = []
        
        # Feature-based analysis
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        prev_descriptors = None
        prev_keypoints = None
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                if prev_descriptors is not None and descriptors is not None:
                    matches = bf.match(prev_descriptors, descriptors)
                    matches = sorted(matches, key=lambda x: x.distance)
                    if len(matches) > 10:
                        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if H is not None:
                            scale = np.sqrt(np.linalg.det(H[0:2, 0:2]))
                            if 0.5 < scale < 2.0:  # Filter extreme values
                                scale_factors.append(scale)
                prev_keypoints, prev_descriptors = keypoints, descriptors
                frame_indices.append(frame_count)
            frame_count += 1
        cap.release()

        # Analyze scale factors
        scale_factors = np.array(scale_factors) if scale_factors else np.array([1.0])
        smoothed_scales = np.convolve(scale_factors, np.ones(5)/5, mode='valid') if len(scale_factors) >= 5 else scale_factors
        avg_zoom = float(np.mean(smoothed_scales))
        min_zoom = float(np.min(scale_factors))
        max_zoom = float(np.max(scale_factors))
        zoom_detected = abs(avg_zoom - 1.0) > 0.1 and len(scale_factors) > int(fps / 2)

        return {
            'avg_zoom': avg_zoom,
            'min_zoom': min_zoom,
            'max_zoom': max_zoom,
            'zoom_detected': zoom_detected
        }
    except Exception as e:
        print(f"Error analyzing zoom in {video_path}: {e}")
        return {
            'error': str(e),
            'avg_zoom': 1.0,
            'min_zoom': 1.0,
            'max_zoom': 1.0,
            'zoom_detected': False
        }

def batch_process_videos(input_path, output_csv, sample_rate=5, max_workers=4, file_extensions=None):
    """
    Process multiple videos in a directory or a single video file.
    
    Args:
        input_path (str): Path to a video file or directory containing videos
        output_csv (str): Path for output CSV file
        sample_rate (int): Sample every nth frame
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
            result = analyze_zoom(video_path, sample_rate=sample_rate)
            print(f"Processed: {os.path.basename(video_path)} - Zoom Score: {result['avg_zoom']:.4f}")
            return {
                "file_name": video_path,
                "zoom_score": result['avg_zoom'],
                "zoom_detected": result['zoom_detected'],
                "min_zoom": result['min_zoom'],
                "max_zoom": result['max_zoom']
            }
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {
                "file_name": video_path,
                "zoom_score": 1.0,
                "zoom_detected": False,
                "min_zoom": 1.0,
                "max_zoom": 1.0,
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
    
    # Create a simplified DataFrame with just filename and zoom score
    simple_df = pd.DataFrame({
        "file_name": results_df["file_name"],
        "zoom_score": results_df["zoom_score"]
    })
    
    # Save results to CSV
    simple_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary
    valid_results = results_df.dropna(subset=['zoom_score'])
    if not valid_results.empty:
        print("\n--- Summary Statistics ---")
        print(f"Total videos processed: {len(results_df)}")
        print(f"Videos successfully analyzed: {len(valid_results)}")
        print(f"Average zoom score: {valid_results['zoom_score'].mean():.4f}")
        print(f"Videos with detected zoom: {sum(valid_results['zoom_detected'])}")
        
    return results_df

if __name__ == "__main__":
    # Set your paths directly in the code
    input_path = "D:\IDARA TEST/Nouveau dossier"  # Path to folder containing videos
    output_csv = "D:\IDARA TEST/zoom_scores_Nouveau_dossier.csv"  # Path for output CSV file
    
    # Parameters
    sample_rate = 5  # Sample every 5th frame
    max_workers = 4
    file_extensions = ['.mp4', '.mpv', '.avi', '.mov', '.MOV']
    
    print(f"Input path: {input_path}")
    print(f"Output CSV: {output_csv}")
    
    results = batch_process_videos(
        input_path=input_path,
        output_csv=output_csv,
        sample_rate=sample_rate,
        max_workers=max_workers,
        file_extensions=file_extensions
    )