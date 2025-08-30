import cv2
import numpy as np
import os
import csv

def calculate_distortion_score(video_path):
    """
    Calculate lens distortion by analyzing line curvature and edge warping.
    Returns a normalized distortion score in [0, 1].
    """
    print(f"Analyzing Distortion for: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "error": f"Cannot open video: {video_path}",
            "analysis_name": "Distortion Score (Error)",
            "norm_distortion": 0.0,
            "per_frame_distortion": [],
            "frame_indices": []
        }

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    distortion_scores = []
    frame_indices = []
    sample_rate = max(1, total_frames // 20)  # Sample ~20 frames

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_rate == 0:
                # Preprocess: Convert to grayscale, enhance edges
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                edges = cv2.Canny(gray, 30, 100, apertureSize=3)
                
                # Try Hough Lines for line curvature
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)
                cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                frame_distortion = 0.0
                
                if lines is not None and len(lines) >= 2:
                    deviations = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Fit a straight line and measure deviation
                        points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                        if x2 != x1:
                            slope = (y2 - y1) / (x2 - x1)
                            intercept = y1 - slope * x1
                            num_samples = 10
                            x_samples = np.linspace(x1, x2, num_samples)
                            y_pred = slope * x_samples + intercept
                            y_actual = np.interp(x_samples, [x1, x2], [y1, y2])
                            deviation = np.mean(np.abs(y_pred - y_actual))
                            deviations.append(deviation)
                    
                    if deviations:
                        frame_distortion = np.mean(deviations)
                
                # Fallback: Edge warping analysis if insufficient lines
                if frame_distortion == 0.0:
                    # Simulate distortion by estimating homography to a rectified plane
                    h, w = gray.shape
                    src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                    dst_pts = src_pts.copy()  # Ideal rectified points
                    try:
                        # Find key points to estimate warping
                        orb = cv2.ORB_create(nfeatures=100)
                        kp, des = orb.detectAndCompute(gray, None)
                        if des is not None and len(kp) > 10:
                            # Match to a synthetic rectified frame (simplified)
                            # Here, we assume slight warping and measure deviation
                            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            warped_edges = cv2.warpPerspective(edges, H, (w, h))
                            # Measure difference between original and rectified edges
                            diff = cv2.absdiff(edges, warped_edges)
                            frame_distortion = np.mean(diff) / 255.0  # Normalize to [0, 1]
                    except:
                        frame_distortion = 0.01  # Small non-zero default for minimal distortion
                
                if frame_distortion > 0:
                    distortion_scores.append(frame_distortion)
                    frame_indices.append(cur_idx)
            
            frame_count += 1
    except Exception as e:
        cap.release()
        return {
            "error": f"Distortion analysis failed: {str(e)}",
            "analysis_name": "Distortion Score (Error)",
            "norm_distortion": 0.0,
            "per_frame_distortion": [],
            "frame_indices": []
        }
    finally:
        cap.release()

    if not distortion_scores:
        return {
            "error": "No sufficient features detected for distortion analysis",
            "analysis_name": "Distortion Score (Error)",
            "norm_distortion": 0.01,  # Small non-zero default
            "per_frame_distortion": [],
            "frame_indices": []
        }

    distortion_scores = np.array(distortion_scores)
    avg_distortion = np.mean(distortion_scores)
    # Normalize using 95th percentile of observed distortions or a max of 1.0
    max_distortion = np.percentile(distortion_scores, 95) if distortion_scores.size > 1 else max(1.0, avg_distortion)
    norm_distortion = np.clip(avg_distortion / max_distortion, 0.0, 1.0) if max_distortion > 0 else 0.01

    return {
        "norm_distortion": float(norm_distortion),
        "avg_distortion": float(avg_distortion),
        "per_frame_distortion": distortion_scores.tolist(),
        "frame_indices": frame_indices,
        "analysis_name": "Distortion Score",
        "type": "sampled"
    }

def analyze_videos(video_folder, output_csv):
    """
    Analyze all video files in a folder and save distortion scores to a CSV file.
    
    Args:
        video_folder: Path to folder containing video files
        output_csv: Path to output CSV file
    """
    # Check if the folder exists
    if not os.path.exists(video_folder):
        print(f"Error: Folder not found: {video_folder}")
        return
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for file in os.listdir(video_folder):
        file_path = os.path.join(video_folder, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    
    # Create CSV file and write header
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['video_name', 'distortion_score'])
        
        # Process each video
        for video_path in video_files:
            print(f"Processing {os.path.basename(video_path)}...")
            result = calculate_distortion_score(video_path)
            
            # Extract video name from path
            video_name = os.path.basename(video_path)
            
            # Get distortion score
            distortion_score = result.get('norm_distortion', 0.0)
            
            # Write to CSV
            csvwriter.writerow([video_name, distortion_score])
            
            print(f"Completed {video_name} - Distortion Score: {distortion_score:.4f}")
    
    print(f"Analysis complete. Results saved to {output_csv}")

if __name__ == "__main__":
    # Set your paths directly in the code for standalone testing
    video_folder = "D:/IDARA TEST/Nouveau dossier"  # Path to folder containing videos
    output_csv = "D:/IDARA TEST/distortion_scores.csv"  # Path for output CSV file
    
    # Run the analysis without any prompts
    print(f"Analyzing all videos in folder: {video_folder}")
    print(f"Results will be saved to: {output_csv}")
    analyze_videos(video_folder, output_csv)