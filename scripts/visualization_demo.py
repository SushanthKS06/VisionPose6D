import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from visualization.pose_visualizer import PoseVisualizer
from aruco_tracking.aruco_detector import ArUcoDetector
from calibration.camera_calibrator import CameraCalibrator
from pose_estimation.pose_estimator import PoseEstimator

def create_synthetic_pose_sequence(num_frames: int = 50) -> list:
    """
    Create a synthetic sequence of poses for demonstration.
    
    Args:
        num_frames: Number of frames to generate
        
    Returns:
        List of pose dictionaries
    """
    poses = []
    
    for t in range(num_frames):
        # Circular motion in XY plane with varying Z
        radius = 200.0  # mm
        height = 300.0 + 50.0 * np.sin(2 * np.pi * t / num_frames)
        
        x = radius * np.cos(2 * np.pi * t / num_frames)
        y = radius * np.sin(2 * np.pi * t / num_frames)
        z = height
        
        # Rotation around Z axis
        angle = 2 * np.pi * t / num_frames
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        rvec = cv2.Rodrigues(rotation_matrix)[0]
        tvec = np.array([[x], [y], [z]], dtype=np.float32)
        
        poses.append({
            'rvec': rvec,
            'tvec': tvec,
            'frame': t
        })
    
    return poses

def demonstrate_static_visualization(visualizer: PoseVisualizer,
                                   detector: ArUcoDetector,
                                   image_path: str) -> None:
    """
    Demonstrate static image visualization capabilities.
    
    Args:
        visualizer: Pose visualizer instance
        detector: ArUco detector instance
        image_path: Path to test image
    """
    print(f"\nStatic Visualization Demo: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Detect markers
    detection_results = detector.detect_markers(
        image, visualizer.camera_matrix, visualizer.dist_coeffs
    )
    
    if detection_results['num_markers'] == 0:
        print("No markers detected in image")
        return
    
    # Process each detected marker
    visualizations = []
    
    for detection in detection_results['detections']:
        marker_id = detection['id']
        
        if 'pose_valid' in detection and detection['pose_valid']:
            rvec = detection['rvec']
            tvec = detection['tvec']
            
            print(f"\nMarker {marker_id} Visualization:")
            
            # Create different visualization modes
            modes = [
                ("Axes Only", lambda img: visualizer.draw_coordinate_axes(img, rvec, tvec)),
                ("Cube Only", lambda img: visualizer.draw_cube(img, rvec, tvec)),
                ("Filled Cube", lambda img: visualizer.draw_cube(img, rvec, tvec, filled=True)),
                ("Comprehensive", lambda img: visualizer.draw_comprehensive_visualization(
                    img, rvec, tvec, marker_id))
            ]
            
            mode_images = []
            mode_titles = []
            
            for title, draw_func in modes:
                mode_img = draw_func(image.copy())
                mode_images.append(mode_img)
                mode_titles.append(title)
                print(f"   Created {title}")
            
            # Create visualization grid
            if len(mode_images) >= 2:
                grid_image = visualizer.create_visualization_grid(
                    mode_images[:4], mode_titles[:4], (2, 2)
                )
                
                # Save grid
                output_path = f"visualization_grid_marker_{marker_id}.jpg"
                cv2.imwrite(output_path, grid_image)
                print(f"   Saved grid to {output_path}")
                
                # Display
                cv2.imshow(f"Marker {marker_id} Visualization Grid", grid_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demonstrate_trajectory_visualization(visualizer: PoseVisualizer) -> None:
    """
    Demonstrate trajectory visualization with synthetic poses.
    
    Args:
        visualizer: Pose visualizer instance
    """
    print(f"\nTrajectory Visualization Demo")
    print("=" * 60)
    
    # Create synthetic pose sequence
    poses = create_synthetic_pose_sequence(num_frames=50)
    
    print(f"Generated {len(poses)} synthetic poses")
    
    # Create 3D trajectory plot
    print("Creating 3D trajectory plot...")
    fig = visualizer.create_3d_plot(poses, "Synthetic Circular Trajectory")
    
    # Save 3D plot
    plot_path = "trajectory_3d_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" Saved 3D plot to {plot_path}")
    
    # Demonstrate 2D trajectory on image
    print("Creating 2D trajectory visualization...")
    
    # Create blank image for trajectory demo
    demo_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw trajectory progressively
    trajectory_images = []
    
    for i, pose in enumerate(poses[::5]):  # Every 5th frame
        img_copy = demo_image.copy()
        
        # Draw axes and cube
        img_copy = visualizer.draw_coordinate_axes(img_copy, pose['rvec'], pose['tvec'])
        img_copy = visualizer.draw_cube(img_copy, pose['rvec'], pose['tvec'])
        img_copy = visualizer.draw_trajectory(img_copy, pose['tvec'])
        img_copy = visualizer.draw_pose_info(img_copy, pose['rvec'], pose['tvec'], 
                                            frame_id=i, position=(10, 30))
        
        trajectory_images.append(img_copy)
    
    # Create trajectory grid
    if trajectory_images:
        trajectory_grid = visualizer.create_visualization_grid(
            trajectory_images[:4], 
            [f"Frame {i*5}" for i in range(len(trajectory_images[:4]))],
            (2, 2)
        )
        
        # Save trajectory grid
        trajectory_path = "trajectory_2d_grid.jpg"
        cv2.imwrite(trajectory_path, trajectory_grid)
        print(f" Saved 2D trajectory grid to {trajectory_path}")
        
        cv2.imshow("2D Trajectory Progression", trajectory_grid)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demonstrate_live_visualization(visualizer: PoseVisualizer,
                                detector: ArUcoDetector,
                                camera_id: int = 0) -> None:
    """
    Demonstrate live pose visualization with camera feed.
    
    Args:
        visualizer: Pose visualizer instance
        detector: ArUco detector instance
        camera_id: Camera device ID
    """
    print(f"\nLive Visualization Demo (Camera {camera_id})")
    print("=" * 60)
    print("Controls:")
    print("  1: Axes only")
    print("  2: Cube only")
    print("  3: Filled cube")
    print("  4: Comprehensive")
    print("  T: Toggle trajectory")
    print("  C: Clear trajectory")
    print("  SPACE: Save frame")
    print("  Q: Quit")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Visualization modes
    visualizer_mode = 4  # Comprehensive by default
    show_trajectory = True
    
    # Pose history for 3D plotting
    pose_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers
        detection_results = detector.detect_markers(
            frame, visualizer.camera_matrix, visualizer.dist_coeffs
        )
        
        # Visualize poses
        result_frame = frame.copy()
        
        for detection in detection_results['detections']:
            if 'pose_valid' in detection and detection['pose_valid']:
                marker_id = detection['id']
                rvec = detection['rvec']
                tvec = detection['tvec']
                
                # Apply visualization based on mode
                if visualizer_mode == 1:
                    result_frame = visualizer.draw_coordinate_axes(result_frame, rvec, tvec)
                elif visualizer_mode == 2:
                    result_frame = visualizer.draw_cube(result_frame, rvec, tvec)
                elif visualizer_mode == 3:
                    result_frame = visualizer.draw_cube(result_frame, rvec, tvec, filled=True)
                elif visualizer_mode == 4:
                    result_frame = visualizer.draw_comprehensive_visualization(
                        result_frame, rvec, tvec, marker_id,
                        show_axes=True, show_cube=True, 
                        show_trajectory=show_trajectory, show_info=True
                    )
                
                # Add to pose history
                pose_history.append({
                    'rvec': rvec.copy(),
                    'tvec': tvec.copy(),
                    'marker_id': marker_id
                })
                
                # Limit history size
                if len(pose_history) > 100:
                    pose_history.pop(0)
        
        # Add mode indicator
        mode_names = {1: "Axes", 2: "Cube", 3: "Filled Cube", 4: "Comprehensive"}
        mode_text = f"Mode: {mode_names.get(visualizer_mode, 'Unknown')} | "
        mode_text += f"Trajectory: {'ON' if show_trajectory else 'OFF'} | "
        mode_text += f"Markers: {detection_results['num_markers']}"
        
        cv2.putText(result_frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add controls
        cv2.putText(result_frame, "1-4: Mode | T: Traj | C: Clear | SPACE: Save | Q: Quit", 
                   (10, result_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Live Pose Visualization", result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            visualizer_mode = int(chr(key))
            print(f"Switched to mode: {mode_names[visualizer_mode]}")
        
        elif key == ord('t'):
            show_trajectory = not show_trajectory
            print(f"Trajectory: {'ON' if show_trajectory else 'OFF'}")
        
        elif key == ord('c'):
            visualizer.clear_trajectory()
            pose_history.clear()
            print("Trajectory cleared")
        
        elif key == ord(' '):
            # Save current frame
            timestamp = cv2.getTickCount()
            output_file = f"live_visualization_{timestamp}.jpg"
            cv2.imwrite(output_file, result_frame)
            print(f"Frame saved: {output_file}")
            
            # Save 3D trajectory if we have enough poses
            if len(pose_history) > 10:
                trajectory_file = f"live_trajectory_{timestamp}.png"
                visualizer.save_trajectory_plot(pose_history, trajectory_file)
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final 3D trajectory plot
    if len(pose_history) > 5:
        print(f"\nSaving final 3D trajectory with {len(pose_history)} poses...")
        visualizer.save_trajectory_plot(pose_history, "final_live_trajectory.png")

def demonstrate_comparison_visualization(visualizer: PoseVisualizer,
                                      detector: ArUcoDetector,
                                      estimator: PoseEstimator,
                                      image_path: str) -> None:
    """
    Demonstrate comparison of different pose estimation algorithms.
    
    Args:
        visualizer: Pose visualizer instance
        detector: ArUco detector instance
        estimator: Pose estimator instance
        image_path: Path to test image
    """
    print(f"\nAlgorithm Comparison Visualization: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Detect markers
    detection_results = detector.detect_markers(
        image, visualizer.camera_matrix, visualizer.dist_coeffs
    )
    
    if detection_results['num_markers'] == 0:
        print("No markers detected in image")
        return
    
    # Process first detected marker
    detection = detection_results['detections'][0]
    marker_id = detection['id']
    
    if 'pose_valid' not in detection or not detection['pose_valid']:
        print("Pose estimation failed for marker")
        return
    
    # Get object and image points
    object_points = detector.marker_object_points
    image_points = detection['corners']
    
    # Test multiple algorithms
    algorithms = ["SOLVEPNP_EPNP", "SOLVEPNP_P3P", "SOLVEPNP_ITERATIVE"]
    comparison_images = []
    algorithm_names = []
    
    print(f"Testing algorithms for marker {marker_id}:")
    
    for algorithm in algorithms:
        try:
            result = estimator.estimate_pose(
                object_points, image_points,
                algorithm=algorithm,
                use_ransac=False,
                refine=True
            )
            
            if result.success:
                # Create visualization
                img_copy = image.copy()
                img_copy = visualizer.draw_comprehensive_visualization(
                    img_copy, result.rotation_vector, result.translation_vector,
                    marker_id, show_axes=True, show_cube=True, show_info=True
                )
                
                # Add algorithm info
                info_text = f"{algorithm}\nError: {result.reprojection_error:.2f}px\nConf: {result.confidence:.2f}"
                cv2.putText(img_copy, info_text, (10, img_copy.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                comparison_images.append(img_copy)
                algorithm_names.append(f"{algorithm}\n({result.reprojection_error:.2f}px)")
                
                print(f"   {algorithm}: Error={result.reprojection_error:.2f}px, Conf={result.confidence:.2f}")
            else:
                print(f"   {algorithm}: Failed")
        
        except Exception as e:
            print(f"   {algorithm}: {e}")
    
    # Also test RANSAC
    try:
        ransac_result = estimator.estimate_pose(
            object_points, image_points,
            use_ransac=True,
            refine=True
        )
        
        if ransac_result.success:
            img_copy = image.copy()
            img_copy = visualizer.draw_comprehensive_visualization(
                img_copy, ransac_result.rotation_vector, ransac_result.translation_vector,
                marker_id, show_axes=True, show_cube=True, show_info=True
            )
            
            info_text = f"RANSAC\nError: {ransac_result.reprojection_error:.2f}px\nInliers: {ransac_result.num_inliers}"
            cv2.putText(img_copy, info_text, (10, img_copy.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            comparison_images.append(img_copy)
            algorithm_names.append(f"RANSAC\n({ransac_result.reprojection_error:.2f}px)")
            
            print(f"   RANSAC: Error={ransac_result.reprojection_error:.2f}px, Inliers={ransac_result.num_inliers}")
    
    except Exception as e:
        print(f"   RANSAC: {e}")
    
    # Create comparison grid
    if len(comparison_images) > 1:
        # Determine grid size
        n_images = len(comparison_images)
        if n_images <= 2:
            rows, cols = 1, 2
        elif n_images <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        # Create grid
        comparison_grid = visualizer.create_visualization_grid(
            comparison_images[:rows*cols], algorithm_names[:rows*cols], (rows, cols)
        )
        
        # Save comparison
        output_path = f"algorithm_comparison_marker_{marker_id}.jpg"
        cv2.imwrite(output_path, comparison_grid)
        print(f"\n Algorithm comparison saved to {output_path}")
        
        cv2.imshow("Algorithm Comparison", comparison_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def load_calibration(calibration_path: str) -> tuple:
    """Load camera calibration parameters."""
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    
    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(calibration_path):
        raise ValueError("Failed to load calibration")
    
    return calibrator.camera_matrix, calibrator.dist_coeffs

def main():
    parser = argparse.ArgumentParser(description="Pose Visualization Demo")
    parser.add_argument('--calibration', required=True, help='Camera calibration file')
    parser.add_argument('--image', type=str, help='Test image for static visualization')
    parser.add_argument('--camera', type=int, help='Camera ID for live visualization')
    parser.add_argument('--trajectory', action='store_true, help='Demonstrate trajectory visualization')
    parser.add_argument('--comparison', action='store_true, help='Demonstrate algorithm comparison')
    
    args = parser.parse_args()
    
    # Load calibration
    try:
        camera_matrix, dist_coeffs = load_calibration(args.calibration)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return
    
    # Initialize components
    visualizer = PoseVisualizer(camera_matrix, dist_coeffs)
    detector = ArUcoDetector()
    estimator = PoseEstimator(camera_matrix, dist_coeffs)
    
    print("Pose Visualization Demonstration")
    print("=" * 60)
    
    if args.image:
        # Static image visualization
        demonstrate_static_visualization(visualizer, detector, args.image)
        
        if args.comparison:
            demonstrate_comparison_visualization(visualizer, detector, estimator, args.image)
    
    elif args.camera is not None:
        # Live visualization
        demonstrate_live_visualization(visualizer, detector, args.camera)
    
    elif args.trajectory:
        # Trajectory demonstration
        demonstrate_trajectory_visualization(visualizer)
    
    else:
        print("Error: Must specify one of --image, --camera, or --trajectory")
        print("Use --help for usage information")
        
        # Default to trajectory demo
        print("\nRunning default trajectory demonstration...")
        demonstrate_trajectory_visualization(visualizer)
    
    print("\nVisualization demonstration complete!")

if __name__ == "__main__":
    main()
