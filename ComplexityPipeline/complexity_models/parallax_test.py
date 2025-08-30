import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from tqdm import tqdm
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
required_packages = ["tqdm", "pandas", "matplotlib"]
for package in required_packages:
    try:
        if package == "tqdm":
            from tqdm import tqdm
            print("tqdm package is already installed")
        elif package == "pandas":
            import pandas
            print("pandas package is already installed")
        elif package == "matplotlib":
            import matplotlib
            print("matplotlib package is already installed")
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

class AdvancedFeatureTracker:
    def __init__(self, video_path, output_path=None, max_features=1000, save_output=False):
        self.video_path = video_path
        self.output_path = output_path or 'tracked_' + os.path.basename(video_path)
        self.max_features = max_features
        self.save_output = save_output
       
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
       
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=1e-4
        )
       
        # History for analysis
        self.displacement_history = []
        self.feature_age = []
        self.avg_flow_history = deque(maxlen=30)  # recent average displacement history
       
        # For depth estimation (if needed later)
        self.camera_matrix = None
        self.depth_map = None
        self.baseline = 1.0  # assumed baseline (scale factor)
       
        # RANSAC parameters for outlier filtering
        self.ransac_threshold = 3.0
        self.min_inliers = 8

    def initialize_video(self):
        """Set up video capture and video writer."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
           
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError("Failed to open video")
           
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       
        # Set up camera intrinsic matrix (estimated based on frame size)
        focal_length = max(self.width, self.height)
        center_x = self.width / 2
        center_y = self.height / 2
        self.camera_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
       
        # Initialize video writer for output video only if save_output is True
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )
       
        # Read the first frame
        ret, self.prev_frame = self.cap.read()
        if not ret:
            raise IOError("Failed to read the first frame")
           
        self.prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        return True

    def detect_features(self, frame_gray):
        """Detect good features to track using Shi-Tomasi method."""
        corners = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        return corners if corners is not None else np.array([])

    def filter_outliers_with_ransac(self, prev_points, current_points):
        """Use RANSAC to filter out motion outliers via fundamental matrix estimation."""
        if len(prev_points) < self.min_inliers:
            return np.array([]), np.array([])
        try:
            result = cv2.findFundamentalMat(
                prev_points, current_points, cv2.FM_RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                confidence=0.99
            )
            if len(result) == 3:
                F, mask, _ = result
            elif len(result) == 2:
                F, mask = result
            else:
                return np.array([]), np.array([])
            if F is None or mask is None:
                return np.array([]), np.array([])
            if isinstance(mask, list):
                mask = mask[0]
            mask = mask.ravel().astype(bool)
            return prev_points[mask], current_points[mask]
        except Exception as e:
            print(f"RANSAC error: {e}")
            return np.array([]), np.array([])

    def estimate_camera_motion(self, prev_points, next_points):
        """Estimate camera rotation (R) and translation (t) from point correspondences."""
        if len(prev_points) < 8:
            return None, None, 0
        try:
            E, mask = cv2.findEssentialMat(
                prev_points, next_points, self.camera_matrix,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if E is None:
                return None, None, 0
            try:
                points_count, R, t, mask = cv2.recoverPose(
                    E, prev_points, next_points, self.camera_matrix, mask=mask
                )
            except ValueError:
                R, t, mask = cv2.recoverPose(
                    E, prev_points, next_points, self.camera_matrix, mask=mask
                )
                points_count = np.sum(mask)
            translation_magnitude = np.linalg.norm(t)
            return R, t, translation_magnitude
        except cv2.error:
            return None, None, 0

    def estimate_relative_depth(self, prev_points, next_points, R, t):
        """Estimate a relative depth score per point (higher score means closer)."""
        if R is None or t is None or len(prev_points) < 4:
            return np.ones(len(prev_points))
        
        prev_points_flat = prev_points.reshape(-1, 2)
        next_points_flat = next_points.reshape(-1, 2)
        displacement_vectors = next_points_flat - prev_points_flat
        magnitudes = np.linalg.norm(displacement_vectors, axis=1)
        directions = np.zeros_like(displacement_vectors)
        non_zero = magnitudes > 0
        if np.any(non_zero):
            for i in range(len(displacement_vectors)):
                if magnitudes[i] > 0:
                    directions[i] = displacement_vectors[i] / magnitudes[i]
        t_direction = t[:2, 0] / np.linalg.norm(t[:2, 0]) if t.shape[1] > 0 else np.array([1, 0])
        alignments = np.abs(np.dot(directions, t_direction))
        depth_scores = alignments / (magnitudes + 0.001)
        if len(depth_scores) > 0 and max(depth_scores) > min(depth_scores):
            normalized_depths = (depth_scores - min(depth_scores)) / (max(depth_scores) - min(depth_scores))
        else:
            normalized_depths = np.ones_like(depth_scores)
        return normalized_depths

    def create_visualization(self, frame, prev_points, current_points, feature_ages=None, depths=None):
        """Visualize tracked feature points with motion vectors and depth coloring."""
        vis_frame = frame.copy()
        depth_overlay = np.zeros_like(frame)
        if len(prev_points) > 0 and len(current_points) > 0:
            min_len = min(len(prev_points), len(current_points))
            prev_points = prev_points[:min_len]
            current_points = current_points[:min_len]
            
            if feature_ages is not None:
                if len(feature_ages) != min_len:
                    feature_ages = (np.ones(min_len) if len(feature_ages) == 0 
                                    else feature_ages[:min_len] if len(feature_ages) > min_len 
                                    else np.pad(feature_ages, (0, min_len - len(feature_ages)), 'constant', constant_values=1))
            else:
                feature_ages = np.ones(min_len)
                
            if depths is not None:
                if len(depths) != min_len:
                    depths = (np.ones(min_len) if len(depths) == 0 
                              else depths[:min_len] if len(depths) > min_len 
                              else np.pad(depths, (0, min_len - len(depths)), 'constant', constant_values=1))
            
            for i in range(min_len):
                prev = prev_points[i]
                curr = current_points[i]
                if hasattr(prev, 'shape') and len(prev.shape) > 1:
                    px, py = prev.flatten()[:2]
                else:
                    px, py = prev
                if hasattr(curr, 'shape') and len(curr.shape) > 1:
                    cx, cy = curr.flatten()[:2]
                else:
                    cx, cy = curr
                displacement = np.sqrt((cx - px)**2 + (cy - py)**2)
                age_factor = min(1.0, feature_ages[i] / 30.0)
                if depths is not None:
                    depth = depths[i]
                    color = (int(255 * (1 - depth)), 50, int(255 * depth))
                    cv2.circle(depth_overlay, (int(cx), int(cy)),
                               radius=max(3, int(15 * depth)),
                               color=color, thickness=-1)
                else:
                    intensity = min(255, int(displacement * 10))
                    color = (0, 255 - intensity, intensity)
                cv2.circle(vis_frame, (int(cx), int(cy)), 3, color, -1)
                cv2.line(vis_frame, (int(px), int(py)), (int(cx), int(cy)),
                         color, max(1, int(2 * age_factor)))
        if len(prev_points) > 0 and depths is not None:
            vis_frame = cv2.addWeighted(vis_frame, 0.7, depth_overlay, 0.3, 0)
        return vis_frame

    def add_info_overlay(self, frame, tracked_count, avg_disp, parallax_score, frame_count):
        """Overlay frame information such as frame number, tracked points, displacement, and parallax score."""
        cv2.rectangle(frame, (10, 10), (300, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 130), (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Tracked Points: {tracked_count}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Mean Displacement: {avg_disp:.3f} px", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Parallax Score: {parallax_score:.3f}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(frame, (self.width - 120, 10), (self.width - 20, 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (self.width - 120, 10), (self.width - 20, 50), (255, 255, 255), 1)
        for i in range(100):
            x = self.width - 120 + i
            color = (int(255 * (i / 100)), 50, int(255 * (1 - i / 100)))
            cv2.line(frame, (x, 15), (x, 45), color, 1)
        cv2.putText(frame, "NEAR", (self.width - 115, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "FAR", (self.width - 50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame

    def process_video(self):
        """Process the entire video to track features, estimate motion, and compute parallax scores."""
        try:
            self.initialize_video()
        except Exception as e:
            print(f"Initialization error: {e}")
            return 0.0
       
        # Detect initial features
        self.prev_points = self.detect_features(self.prev_gray)
        self.feature_age = np.zeros(len(self.prev_points))
       
        frame_count = 0
        parallax_scores = []
       
        while self.cap.isOpened():
            ret, current_frame = self.cap.read()
            if not ret:
                break
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
           
            if len(self.prev_points) > 0:
                next_points, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, current_gray, self.prev_points, None, **self.lk_params
                )
                if next_points is not None:
                    status = status.ravel()
                    good_old = self.prev_points[status == 1]
                    good_new = next_points[status == 1]
                   
                    if len(self.feature_age) == len(status):
                        good_age = self.feature_age[status == 1] + 1
                    else:
                        good_age = np.ones(len(good_old))
                   
                    if len(good_old) >= self.min_inliers:
                        filtered_old, filtered_new = self.filter_outliers_with_ransac(good_old, good_new)
                        if len(filtered_old) > 0:
                            old_points_tuple = [tuple(pt.flatten()) for pt in good_old]
                            filtered_points_tuple = [tuple(pt.flatten()) for pt in filtered_old]
                            filtered_age = []
                            for pt in filtered_points_tuple:
                                if pt in old_points_tuple:
                                    idx = old_points_tuple.index(pt)
                                    filtered_age.append(good_age[idx])
                                else:
                                    filtered_age.append(1)
                            good_old = filtered_old
                            good_new = filtered_new
                            good_age = np.array(filtered_age)
                   
                    if len(good_new) > 0:
                        displacement = good_new - good_old
                        displacements = np.linalg.norm(displacement, axis=1)
                        avg_disp = np.mean(displacements)
                        self.displacement_history.append(avg_disp)
                        self.avg_flow_history.append(avg_disp)
                        smoothed_disp = np.mean(list(self.avg_flow_history))
                       
                        R, t, trans_magnitude = self.estimate_camera_motion(good_old, good_new)
                       
                        disp_variance = np.var(displacements) if len(displacements) > 1 else 0
                        parallax_score = smoothed_disp * (1 + disp_variance)
                        if trans_magnitude > 0:
                            parallax_score *= trans_magnitude
                        parallax_scores.append(parallax_score)
                       
                        depths = self.estimate_relative_depth(good_old, good_new, R, t)
                    else:
                        avg_disp = 0
                        self.displacement_history.append(0)
                        parallax_score = 0
                        parallax_scores.append(0)
                        depths = None
                   
                    if self.save_output:
                        visual_frame = self.create_visualization(
                            current_frame, good_old, good_new, good_age, depths
                        )
                        
                        visual_frame = self.add_info_overlay(
                            visual_frame, len(good_new), avg_disp, parallax_score, frame_count
                        )
                        
                        cv2.imshow('Advanced Feature Tracking', visual_frame)
                        self.out.write(visual_frame)
                   
                    self.prev_points = good_new.reshape(-1, 1, 2)
                    self.feature_age = good_age
                else:
                    self.prev_points = np.array([])
                    self.feature_age = np.array([])
           
            if len(self.prev_points) < 0.5 * self.max_features:
                new_features = self.detect_features(current_gray)
                if len(new_features) > 0:
                    if len(self.prev_points) > 0:
                        prev_pts = self.prev_points.reshape(-1, 2)
                        new_pts = new_features.reshape(-1, 2)
                        filtered_new = []
                        for pt in new_pts:
                            dists = np.sqrt(np.sum((prev_pts - pt) ** 2, axis=1))
                            if np.all(dists > 10):  # minimum distance threshold
                                filtered_new.append(pt)
                        if filtered_new:
                            filtered_new = np.array(filtered_new).reshape(-1, 1, 2)
                            self.prev_points = np.vstack((self.prev_points, filtered_new))
                            self.feature_age = np.hstack((self.feature_age, np.zeros(len(filtered_new))))
                    else:
                        self.prev_points = new_features
                        self.feature_age = np.zeros(len(new_features))
           
            self.prev_gray = current_gray.copy()
            frame_count += 1
           
            if cv2.waitKey(1) == 27:
                break
       
        self.cap.release()
        if self.save_output:
            self.out.release()
            cv2.destroyAllWindows()
       
        # Calculate overall parallax score
        if len(parallax_scores) > 0:
            min_score = min(parallax_scores)
            max_score = max(parallax_scores)
            if max_score > min_score:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in parallax_scores]
                overall_norm_score = np.mean(normalized_scores)
            else:
                overall_norm_score = 0.0  # Default value when scores don't vary
        else:
            overall_norm_score = 0.0  # No scores available
        
        return overall_norm_score

def calculate_parallax_score(video_path, save_output=False):
    """Helper function to process the video and return a normalized parallax score."""
    try:
        tracker = AdvancedFeatureTracker(video_path, save_output=save_output)
        overall_norm_score = tracker.process_video()
        return overall_norm_score
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return 0.0

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
    
    # Process videos in parallel
    results = []
    if max_workers > 1 and len(video_files) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(executor.map(calculate_parallax_score, video_files))
            for video_path, score in zip(video_files, futures):
                results.append({
                    "file_name": video_path,
                    "parallax_score": score
                })
                print(f"Processed: {os.path.basename(video_path)} - Score: {score:.4f}")
    else:
        for video_path in video_files:
            print(f"Processing: {os.path.basename(video_path)}")
            score = calculate_parallax_score(video_path)
            results.append({
                "file_name": video_path,
                "parallax_score": score
            })
            print(f"Processed: {os.path.basename(video_path)} - Score: {score:.4f}")
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary
    valid_results = results_df.dropna(subset=['parallax_score'])
    if not valid_results.empty:
        print("\n--- Summary Statistics ---")
        print(f"Total videos processed: {len(results_df)}")
        print(f"Videos successfully analyzed: {len(valid_results)}")
        print(f"Average parallax score: {valid_results['parallax_score'].mean():.4f}")
        
    return results_df

if __name__ == "__main__":
    # Set your paths directly in the code
    input_path = "D:/IDARA TEST/V02 - DATA/EASY"  # Path to folder containing videos
    output_csv = "D:\IDARA TEST\parallax_scores_hard_shots.csv"  # Path for output CSV file
    
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


