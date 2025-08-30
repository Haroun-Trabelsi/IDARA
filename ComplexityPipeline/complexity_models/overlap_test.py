import cv2
import numpy as np
import os
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
required_packages = ["lap>=0.5.12", "ultralytics"]
for package in required_packages:
    try:
        if package.startswith("lap"):
            import lap
            print("lap package is already installed")
        elif package.startswith("ultralytics"):
            from ultralytics import YOLO
            print("ultralytics package is already installed")
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

# Now import the required modules
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

def analyze_overlap_complexity(input_path, output_path, conf_threshold=0.5, iou_threshold=0.3,
                              tracker_config="botsort.yaml", mask_type="Black Mask", class_names=None, movement_threshold=None):
    """
    Analyzes video for object overlap complexity and applies masks to tracked objects of specified classes in all frames.
    The overlap complexity score includes both average overlap and the percentage of frames with masks, normalized to 0-1.
    
    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        conf_threshold (float): Confidence threshold for object detection.
        iou_threshold (float): IOU threshold for object detection.
        tracker_config (str): Tracker configuration file (default: botsort.yaml).
        mask_type (str): Type of mask to apply ("Black Mask", "Blur", "Red Overlay").
        class_names (str or list): Class name(s) to mask (e.g., 'person' or ['person', 'car']); if None, mask all classes.
        movement_threshold (float, optional): Ignored parameter for backward compatibility.
        
    Returns:
        dict: Dictionary containing overlap analysis results including overlap score (normalized to 0-1).
    """
    # Load the YOLO model - use a default model (yolov8n-seg)
    try:
        model = YOLO("./yolov8n-seg.pt")
    except Exception as e:
        raise ValueError(f"Failed to load YOLO model: {e}")
    
    # Validate class_names
    valid_classes = model.names if hasattr(model, 'names') else {}
    if class_names:
        # Convert single class_names string to list for uniform handling
        if isinstance(class_names, str):
            class_names = [class_names]
        # Validate each class name
        for cls in class_names:
            if cls not in valid_classes.values():
                raise ValueError(f"Invalid class name '{cls}'. Valid classes: {list(valid_classes.values())}")
    else:
        class_names = list(valid_classes.values())  # Mask all classes if None
    
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_area = frame_width * frame_height
    
    # Create output directories for frames and debug masks
    frames_dir = os.path.join(os.path.dirname(output_path), "masked_frames")
    debug_dir = os.path.join(os.path.dirname(output_path), "debug_masks")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize variables for analytics
    frames_with_masks = 0
    max_masked_area_percent = 0
    max_masked_area_frame_num = 0
    current_frame = 0
    
    # For overlap calculation
    total_overlap_score = 0
    max_overlap_score = 0
    max_overlap_frame = 0
    frames_with_overlap = 0
    
    # Track object positions and masks
    previous_objects = {}  # {track_id: {'centroid': (x, y), 'mask': np.array}}
    previous_masks = []    # List of masks for overlap calculation
    
    print(f"Processing video with {total_frames} frames...")
    
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            break
        
        current_frame += 1
        if current_frame % 10 == 0:
            print(f"Processing frame {current_frame}/{total_frames}")
            
        # Create a mask for this frame
        frame_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        # Run YOLO tracking on the frame
        results = model.track(frame, persist=True, conf=conf_threshold, iou=iou_threshold, tracker=tracker_config)
        
        # Object-specific masks and data for this frame
        current_object_masks = []
        current_objects = {}
        has_masks = False
        
        if results[0].masks is not None and results[0].boxes is not None:
            # Process each detected object
            for i, (segment, box) in enumerate(zip(results[0].masks.xy, results[0].boxes)):
                # Get class and track ID
                cls_id = int(box.cls) if box.cls is not None else None
                track_id = int(box.id) if box.id is not None else None
                if cls_id is None or track_id is None:
                    continue
                
                # Filter by class_names
                if valid_classes.get(cls_id) not in class_names:
                    continue
                
                # Create individual object mask with precise coordinates
                object_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                segment_np = np.array(segment, dtype=np.float32)  # Use float32 for precision
                segment_np = segment_np.round().astype(np.int32)  # Round to nearest integer
                cv2.fillPoly(object_mask, [segment_np], 255)
                
                # Save individual object mask for debugging
                cv2.imwrite(os.path.join(debug_dir, f"frame_{current_frame}_object_{track_id}_mask.jpg"), object_mask)
                
                # Calculate centroid from bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Apply mask for all detected objects
                frame_mask = cv2.bitwise_or(frame_mask, object_mask)
                current_object_masks.append(object_mask)
                has_masks = True
                
                # Store object data
                current_objects[track_id] = {'centroid': centroid, 'mask': object_mask}
        
        # Save combined frame mask for debugging
        if has_masks:
            cv2.imwrite(os.path.join(debug_dir, f"frame_{current_frame}_combined_mask.jpg"), frame_mask)
        
        # Calculate overlap with previous frame
        frame_overlap_score = 0
        if has_masks and previous_masks:
            for curr_mask in current_object_masks:
                for prev_mask in previous_masks:
                    intersection = cv2.bitwise_and(curr_mask, prev_mask)
                    intersection_area = cv2.countNonZero(intersection)
                    if intersection_area > 0:
                        curr_area = cv2.countNonZero(curr_mask)
                        prev_area = cv2.countNonZero(prev_mask)
                        union_area = curr_area + prev_area - intersection_area
                        iou = intersection_area / union_area if union_area > 0 else 0
                        smaller_area = min(curr_area, prev_area)
                        overlap_percentage = intersection_area / smaller_area if smaller_area > 0 else 0
                        frame_overlap_score += (intersection_area / total_area) * overlap_percentage
        
        # Update previous data
        previous_masks = current_object_masks
        previous_objects = current_objects
        
        # Update overlap statistics
        if frame_overlap_score > 0:
            frames_with_overlap += 1
            total_overlap_score += frame_overlap_score
            if frame_overlap_score > max_overlap_score:
                max_overlap_score = frame_overlap_score
                max_overlap_frame = current_frame
        
        if has_masks:
            frames_with_masks += 1
            masked_pixels = cv2.countNonZero(frame_mask)
            masked_area_percent = (masked_pixels / total_area) * 100
            if masked_area_percent > max_masked_area_percent:
                max_masked_area_percent = masked_area_percent
                max_masked_area_frame_num = current_frame
                cv2.imwrite(os.path.join(frames_dir, "max_masked_frame.jpg"), frame)
                cv2.imwrite(os.path.join(frames_dir, "max_masked_frame_mask.jpg"), frame_mask)
            
            # Apply mask based on mask_type
            masked_frame = frame.copy()
            if mask_type == "Black Mask":
                masked_frame[frame_mask > 0] = [0, 0, 0]
            elif mask_type == "Blur":
                # Apply blur only to masked regions
                blurred = cv2.GaussianBlur(frame, (55, 55), 0)
                mask_3d = np.repeat(frame_mask[:, :, np.newaxis], 3, axis=2) / 255.0
                masked_frame = (frame * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
            elif mask_type == "Red Overlay":
                # Apply red overlay with precise alpha blending
                mask_colored = np.zeros_like(frame)
                mask_colored[frame_mask > 0] = [0, 0, 255]  # Red color
                mask_3d = np.repeat(frame_mask[:, :, np.newaxis], 3, axis=2) / 255.0
                masked_frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.5, 0.0)
            
            # Add text with analytics
            cv2.putText(masked_frame, f"Masked: {masked_area_percent:.2f}%", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(masked_frame, f"Overlap: {frame_overlap_score:.4f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(masked_frame, f"Frame: {current_frame}/{total_frames}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(masked_frame)
        else:
            out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calculate final overlap complexity score including frames_with_masks_percent, normalized to 0-1
    average_overlap = total_overlap_score / frames_with_overlap if frames_with_overlap > 0 else 0
    frames_with_masks_percent = (frames_with_masks / current_frame) * 100 if current_frame > 0 else 0
    overlap_complexity_score = min(1.0, (0.5 * (1 - np.exp(-5 * average_overlap)) + 0.5 * (frames_with_masks_percent / 100)))
    
    # Prepare results
    results = {
        "input_video": os.path.basename(input_path),
        "total_frames": current_frame,
        "frames_with_masks": frames_with_masks,
        "frames_with_masks_percent": frames_with_masks_percent,
        "max_masked_area_frame": max_masked_area_frame_num,
        "max_masked_area_percent": max_masked_area_percent,
        "frames_with_overlap": frames_with_overlap,
        "frames_with_overlap_percent": (frames_with_overlap / current_frame) * 100 if current_frame > 0 else 0,
        "max_overlap_frame": max_overlap_frame,
        "max_overlap_score": max_overlap_score,
        "average_overlap": average_overlap,
        "average_overlap_score": average_overlap * 100,  # For cumulative_results_df
        "overlap_complexity_score": overlap_complexity_score,
        "output_video_path": output_path,
        "max_masked_frame_path": os.path.join(frames_dir, "max_masked_frame.jpg") if frames_with_masks > 0 else None,
        "debug_masks_dir": debug_dir
    }
    
    print("\n--- Overlap Complexity Analysis Results ---")
    print(f"Video: {os.path.basename(input_path)}")
    print(f"Overall complexity score: {results['overlap_complexity_score']:.4f}/1")
    print(f"Total frames processed: {results['total_frames']}")
    print(f"Frames with masks: {results['frames_with_masks']} ({results['frames_with_masks_percent']:.2f}%)")
    print(f"Frames with overlap: {results['frames_with_overlap']} ({results['frames_with_overlap_percent']:.2f}%)")
    print(f"Maximum overlap score: {results['max_overlap_score']:.4f} (Frame {results['max_overlap_frame']})")
    print(f"Average overlap score: {results['average_overlap']:.4f}")
    print(f"Processed video saved to: {results['output_video_path']}")
    
    return results

def batch_process_videos(input_path, output_dir, conf_threshold=0.5, iou_threshold=0.3,
                         tracker_config="botsort.yaml", mask_type="Black Mask", class_names=None, 
                         max_workers=4, file_extensions=None):
    """
    Process multiple videos in a directory or a single video file.
    
    Args:
        input_path (str): Path to a video file or directory containing videos
        output_dir (str): Directory for output CSV file
        conf_threshold (float): Confidence threshold for object detection
        iou_threshold (float): IOU threshold for object detection
        tracker_config (str): Tracker configuration file
        mask_type (str): Type of mask to apply
        class_names (str or list): Class name(s) to mask
        max_workers (int): Maximum number of parallel workers
        file_extensions (list): List of file extensions to process (default: ['.mp4', '.mpv', '.avi'])
        
    Returns:
        pd.DataFrame: DataFrame containing results for all processed videos
    """
    if file_extensions is None:
        file_extensions = ['.mp4', '.mpv', '.avi']
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
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
    
    # Define a simplified function to analyze video without generating output files
    def analyze_video_simple(video_path):
        try:
            video_name = os.path.basename(video_path)
            print(f"\nProcessing video: {video_name}")
            
            # Load the YOLO model
            try:
                model = YOLO("yolov8n-seg.pt")
            except Exception as e:
                raise ValueError(f"Failed to load YOLO model: {e}")
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_area = frame_width * frame_height
            
            # Initialize variables for analytics
            frames_with_masks = 0
            current_frame = 0
            total_overlap_score = 0
            frames_with_overlap = 0
            
            # For tracking objects and masks
            previous_masks = []
            
            print(f"Analyzing video with {total_frames} frames...")
            
            # Process every 5th frame to speed up analysis
            frame_step = 5
            
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()
                if not success:
                    break
                
                current_frame += 1
                
                # Skip frames to speed up processing
                if current_frame % frame_step != 0 and current_frame < total_frames - 1:
                    continue
                
                if current_frame % 50 == 0:
                    print(f"Processing frame {current_frame}/{total_frames}")
                
                # Run YOLO tracking on the frame
                results = model.track(frame, persist=True, conf=conf_threshold, iou=iou_threshold, tracker=tracker_config)
                
                # Object-specific masks for this frame
                current_object_masks = []
                has_masks = False
                
                if results[0].masks is not None and results[0].boxes is not None:
                    # Process each detected object
                    for segment, box in zip(results[0].masks.xy, results[0].boxes):
                        # Create individual object mask
                        object_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                        segment_np = np.array(segment, dtype=np.float32)
                        segment_np = segment_np.round().astype(np.int32)
                        cv2.fillPoly(object_mask, [segment_np], 255)
                        
                        current_object_masks.append(object_mask)
                        has_masks = True
                
                # Calculate overlap with previous frame
                frame_overlap_score = 0
                if has_masks and previous_masks:
                    for curr_mask in current_object_masks:
                        for prev_mask in previous_masks:
                            intersection = cv2.bitwise_and(curr_mask, prev_mask)
                            intersection_area = cv2.countNonZero(intersection)
                            if intersection_area > 0:
                                curr_area = cv2.countNonZero(curr_mask)
                                prev_area = cv2.countNonZero(prev_mask)
                                union_area = curr_area + prev_area - intersection_area
                                smaller_area = min(curr_area, prev_area)
                                overlap_percentage = intersection_area / smaller_area if smaller_area > 0 else 0
                                frame_overlap_score += (intersection_area / total_area) * overlap_percentage
                
                # Update previous data
                previous_masks = current_object_masks
                
                # Update statistics
                if frame_overlap_score > 0:
                    frames_with_overlap += 1
                    total_overlap_score += frame_overlap_score
                
                if has_masks:
                    frames_with_masks += 1
            
            # Release resources
            cap.release()
            
            # Calculate final overlap complexity score
            average_overlap = total_overlap_score / frames_with_overlap if frames_with_overlap > 0 else 0
            frames_with_masks_percent = (frames_with_masks / (total_frames // frame_step)) * 100
            overlap_complexity_score = min(1.0, (0.5 * (1 - np.exp(-5 * average_overlap)) + 0.5 * (frames_with_masks_percent / 100)))
            
            print(f"Completed analysis for {video_name} - Overlap score: {overlap_complexity_score:.4f}")
            
            return {
                "file_name": video_path,
                "overlap_score": overlap_complexity_score
            }
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {
                "file_name": video_path,
                "overlap_score": None,
                "error": str(e)
            }
    
    # Process videos in parallel
    results = []
    if max_workers > 1 and len(video_files) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(analyze_video_simple, video_files), total=len(video_files)):
                results.append(result)
    else:
        for video_path in tqdm(video_files):
            result = analyze_video_simple(video_path)
            results.append(result)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(output_dir, index=False)
    print(f"\nResults saved to: {output_dir}")
    
    # Print summary
    valid_results = results_df.dropna(subset=['overlap_score'])
    if not valid_results.empty:
        print("\n--- Summary Statistics ---")
        print(f"Total videos processed: {len(results_df)}")
        print(f"Videos successfully analyzed: {len(valid_results)}")
        print(f"Average complexity score: {valid_results['overlap_score'].mean():.4f}")
        
    return results_df

if __name__ == "__main__":
    # Set your paths directly in the code
    input_path = "D:/IDARA TEST/V02 - DATA/EASY"  # Path to folder containing videos
    output_csv = "D:\IDARA TEST\overlap_scores_prd.csv"  # Path for output CSV file
    
    # Parameters
    conf_threshold = 0.5
    iou_threshold = 0.3
    tracker_config = "botsort.yaml"
    mask_type = "Black Mask"
    class_names = None  # Set to specific classes if needed, e.g. ["person", "car"]
    max_workers = 4
    file_extensions = ['.mp4', '.mpv', '.avi', '.mov', '.MOV']  # Added MOV files
    
    print(f"Input path: {input_path}")
    print(f"Output CSV: {output_csv}")
    
    results = batch_process_videos(
        input_path=input_path,
        output_dir=output_csv,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        tracker_config=tracker_config,
        mask_type=mask_type,
        class_names=class_names,
        max_workers=max_workers,
        file_extensions=file_extensions
    )