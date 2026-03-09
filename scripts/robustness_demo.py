import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from robustness.kalman_filter import PoseKalmanFilter, AdaptivePoseFilter
from pose_estimation.pose_estimator import PoseEstimator
from aruco_tracking.aruco_detector import ArUcoDetector
from calibration.camera_calibrator import CameraCalibrator

def create_noisy_pose_sequence(num_frames: int=100, noise_level: float=5.0, outlier_probability: float=0.05, dropout_probability: float=0.1) -> tuple:
    true_poses = []
    noisy_poses = []
    timestamps = []
    for t in range(num_frames):
        timestamp = t * 0.033
        scale = 200.0
        x = scale * np.sin(2 * np.pi * t / num_frames * 2)
        y = scale * np.sin(2 * np.pi * t / num_frames * 4)
        z = 300.0 + 50.0 * np.sin(2 * np.pi * t / num_frames)
        angle = 2 * np.pi * t / num_frames
        rotation = np.array([0, 0, angle], dtype=np.float32)
        true_position = np.array([[x], [y], [z]], dtype=np.float32)
        true_poses.append((true_position, rotation))
        if np.random.random() > dropout_probability:
            noise = np.random.normal(0, noise_level, (3, 1))
            noisy_position = true_position + noise
            if np.random.random() < outlier_probability:
                outlier = np.random.normal(0, noise_level * 10, (3, 1))
                noisy_position = true_position + outlier
                confidence = 0.1
            else:
                confidence = 0.8
            noisy_poses.append((noisy_position, rotation, confidence))
        else:
            noisy_poses.append((None, None, 0.0))
        timestamps.append(timestamp)
    return (true_poses, noisy_poses, timestamps)

def demonstrate_kalman_filtering():
    print('\n' + '=' * 60)
    print('KALMAN FILTER DEMONSTRATION')
    print('=' * 60)
    true_poses, noisy_poses, timestamps = create_noisy_pose_sequence(num_frames=200, noise_level=10.0, outlier_probability=0.1)
    kf = PoseKalmanFilter(process_noise_position=2.0, process_noise_rotation=0.05, measurement_noise_position=8.0, measurement_noise_rotation=0.3)
    filtered_poses = []
    predicted_poses = []
    for i, (true_pose, noisy_pose) in enumerate(zip(true_poses, noisy_poses)):
        true_pos, true_rot = true_pose
        noisy_pos, noisy_rot, confidence = noisy_pose
        if noisy_pos is not None:
            filtered_pos, filtered_rot = kf.update(noisy_pos, noisy_rot, confidence)
            filtered_poses.append((filtered_pos, filtered_rot))
            predicted_poses.append(None)
        else:
            pred_pos, pred_rot = kf.predict()
            filtered_poses.append((pred_pos, pred_rot))
            predicted_poses.append((pred_pos, pred_rot))
    position_errors = []
    filtered_errors = []
    for i, (true_pose, filtered_pose) in enumerate(zip(true_poses, filtered_poses)):
        true_pos, _ = true_pose
        filtered_pos, _ = filtered_pose
        if i < len(noisy_poses):
            noisy_pos, _, _ = noisy_poses[i]
            if noisy_pos is not None:
                pos_error = np.linalg.norm(true_pos - noisy_pos)
                position_errors.append(pos_error)
        filtered_error = np.linalg.norm(true_pos - filtered_pos)
        filtered_errors.append(filtered_error)
    print(f'Original position error: Mean={np.mean(position_errors):.2f}mm, Std={np.std(position_errors):.2f}mm')
    print(f'Filtered position error: Mean={np.mean(filtered_errors):.2f}mm, Std={np.std(filtered_errors):.2f}mm')
    print(f'Error reduction: {(np.mean(position_errors) - np.mean(filtered_errors)) / np.mean(position_errors) * 100:.1f}%')
    plt.figure(figsize=(15, 5))
    true_positions = np.array([pose[0].flatten() for pose in true_poses])
    filtered_positions = np.array([pose[0].flatten() for pose in filtered_poses])
    plt.subplot(1, 3, 1)
    plt.plot(true_positions[:, 0], 'g-', label='True', linewidth=2)
    plt.plot(filtered_positions[:, 0], 'b-', label='Filtered', linewidth=1)
    plt.title('X Position')
    plt.xlabel('Frame')
    plt.ylabel('Position (mm)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(true_positions[:, 1], 'g-', label='True', linewidth=2)
    plt.plot(filtered_positions[:, 1], 'b-', label='Filtered', linewidth=1)
    plt.title('Y Position')
    plt.xlabel('Frame')
    plt.ylabel('Position (mm)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    if position_errors:
        plt.plot(position_errors, 'r-', label='Original', alpha=0.7)
    plt.plot(filtered_errors, 'b-', label='Filtered', linewidth=2)
    plt.title('Position Error')
    plt.xlabel('Frame')
    plt.ylabel('Error (mm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kalman_filter_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Kalman filter demo plot saved to: kalman_filter_demo.png')

def demonstrate_adaptive_filtering():
    print('\n' + '=' * 60)
    print('ADAPTIVE FILTER DEMONSTRATION')
    print('=' * 60)
    true_poses, noisy_poses, timestamps = create_noisy_pose_sequence(num_frames=150, noise_level=5.0, outlier_probability=0.2, dropout_probability=0.15)
    adaptive_filter = AdaptivePoseFilter()
    filtered_results = []
    for i, (true_pose, noisy_pose) in enumerate(zip(true_poses, noisy_poses)):
        true_pos, true_rot = true_pose
        noisy_pos, noisy_rot, confidence = noisy_pose
        if noisy_pos is not None:
            result = adaptive_filter.process_measurement(noisy_pos, noisy_rot, confidence, timestamps[i])
            filtered_results.append(result)
        elif adaptive_filter.kalman_filter.initialized:
            pred_pos, pred_rot = adaptive_filter.kalman_filter.predict()
            result = {'filtered_position': pred_pos, 'filtered_rotation': pred_rot, 'is_outlier': False, 'is_predicted': True, 'confidence': 0.3}
            filtered_results.append(result)
        else:
            filtered_results.append(None)
    outliers_detected = sum((1 for r in filtered_results if r and r['is_outlier']))
    predictions_made = sum((1 for r in filtered_results if r and r['is_predicted']))
    total_processed = len([r for r in filtered_results if r])
    print(f'Total measurements processed: {total_processed}')
    print(f'Outliers detected: {outliers_detected} ({outliers_detected / total_processed * 100:.1f}%)')
    print(f'Predictions made: {predictions_made} ({predictions_made / total_processed * 100:.1f}%)')
    true_positions = np.array([pose[0].flatten() for pose in true_poses])
    filtered_positions = np.array([r['filtered_position'].flatten() for r in filtered_results if r])
    errors = np.linalg.norm(true_positions[:len(filtered_positions)] - filtered_positions, axis=1)
    print(f'Mean error after filtering: {np.mean(errors):.2f}mm')
    print(f'Error standard deviation: {np.std(errors):.2f}mm')
    stats = adaptive_filter.get_statistics()
    print(f'\nFilter Statistics:')
    print(f"  Rejection rate: {stats['rejection_rate'] * 100:.1f}%")
    print(f"  Prediction rate: {stats['prediction_rate'] * 100:.1f}%")
    print(f"  Update rate: {stats['update_rate'] * 100:.1f}%")
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True', linewidth=2)
    plt.plot(filtered_positions[:, 0], filtered_positions[:, 1], 'b-', label='Filtered', linewidth=1)
    plt.title('XY Trajectory')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.subplot(2, 2, 2)
    plt.plot(errors, 'r-', linewidth=2)
    plt.title('Filtering Error Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Error (mm)')
    plt.grid(True)
    plt.subplot(2, 2, 3)
    confidences = [r['confidence'] for r in filtered_results if r]
    outlier_flags = [r['is_outlier'] for r in filtered_results if r]
    predicted_flags = [r['is_predicted'] for r in filtered_results if r]
    x_axis = range(len(confidences))
    plt.plot(x_axis, confidences, 'b-', label='Confidence', linewidth=2)
    outlier_indices = [i for i, flag in enumerate(outlier_flags) if flag]
    predicted_indices = [i for i, flag in enumerate(predicted_flags) if flag]
    if outlier_indices:
        plt.scatter([x_axis[i] for i in outlier_indices], [confidences[i] for i in outlier_indices], c='red', s=50, label='Outliers', zorder=5)
    if predicted_indices:
        plt.scatter([x_axis[i] for i in predicted_indices], [confidences[i] for i in predicted_indices], c='orange', s=50, label='Predicted', zorder=5)
    plt.title('Measurement Confidence')
    plt.xlabel('Frame')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.hist(errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Error Distribution')
    plt.xlabel('Error (mm)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('adaptive_filter_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Adaptive filter demo plot saved to: adaptive_filter_demo.png')

def demonstrate_real_time_filtering(camera_id: int, calibration_file: str):
    print(f'\n' + '=' * 60)
    print('REAL-TIME FILTERING DEMONSTRATION')
    print('=' * 60)
    print(f'Camera: {camera_id}')
    print(f'Calibration: {calibration_file}')
    print('Controls: SPACE=Toggle filter, C=Clear, Q=Quit')
    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(calibration_file):
        print('Failed to load calibration')
        return
    detector = ArUcoDetector()
    estimator = PoseEstimator(calibrator.camera_matrix, calibrator.dist_coeffs)
    adaptive_filter = AdaptivePoseFilter()
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    use_filter = True
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        detection_results = detector.detect_markers(frame, calibrator.camera_matrix, calibrator.dist_coeffs)
        for detection in detection_results['detections']:
            marker_id = detection['id']
            if 'pose_valid' in detection and detection['pose_valid']:
                object_points = detector.marker_object_points
                image_points = detection['corners']
                pose_result = estimator.estimate_pose(object_points, image_points, use_ransac=True)
                if pose_result.success:
                    position = pose_result.translation_vector
                    rotation = pose_result.rotation_vector
                    confidence = pose_result.confidence
                    if use_filter:
                        filter_result = adaptive_filter.process_measurement(position, rotation, confidence)
                        filtered_pos = filter_result['filtered_position']
                        filtered_rot = filter_result['filtered_rotation']
                        from visualization.pose_visualizer import PoseVisualizer
                        visualizer = PoseVisualizer(calibrator.camera_matrix, calibrator.dist_coeffs)
                        frame = visualizer.draw_comprehensive_visualization(frame, filtered_rot, filtered_pos, marker_id)
                        status = 'FILTERED'
                        if filter_result['is_outlier']:
                            status += ' (OUTLIER)'
                        if filter_result['is_predicted']:
                            status += ' (PREDICTED)'
                        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        from visualization.pose_visualizer import PoseVisualizer
                        visualizer = PoseVisualizer(calibrator.camera_matrix, calibrator.dist_coeffs)
                        frame = visualizer.draw_comprehensive_visualization(frame, rotation, position, marker_id)
                        cv2.putText(frame, 'RAW', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        filter_status = f"Filter: {('ON' if use_filter else 'OFF')}"
        cv2.putText(frame, filter_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'SPACE: Toggle | C: Clear | Q: Quit', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Real-Time Filtering Demo', frame)
        key = cv2.waitKey(1) & 255
        if key == ord(' '):
            use_filter = not use_filter
            print(f"Filter {('enabled' if use_filter else 'disabled')}")
        elif key == ord('c'):
            adaptive_filter.reset()
            print('Filter reset')
        elif key == ord('q'):
            break
    stats = adaptive_filter.get_statistics()
    print(f'\nFinal Statistics:')
    print(f"  Total measurements: {stats['total_measurements']}")
    print(f"  Rejection rate: {stats['rejection_rate'] * 100:.1f}%")
    print(f"  Prediction rate: {stats['prediction_rate'] * 100:.1f}%")
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Robustness Features Demo')
    parser.add_argument('--kalman', action='store_true', help='Demonstrate Kalman filtering')
    parser.add_argument('--adaptive', action='store_true', help='Demonstrate adaptive filtering')
    parser.add_argument('--realtime', action='store_true', help='Real-time demonstration')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for real-time demo')
    parser.add_argument('--calibration', type=str, help='Camera calibration file')
    args = parser.parse_args()
    if args.kalman:
        demonstrate_kalman_filtering()
    elif args.adaptive:
        demonstrate_adaptive_filtering()
    elif args.realtime:
        if not args.calibration:
            print('Error: --calibration required for real-time demo')
            return
        if not Path(args.calibration).exists():
            print(f'Error: Calibration file not found: {args.calibration}')
            return
        demonstrate_real_time_filtering(args.camera, args.calibration)
    else:
        print('Running all demonstrations...')
        demonstrate_kalman_filtering()
        demonstrate_adaptive_filtering()
        if args.calibration and Path(args.calibration).exists():
            demonstrate_real_time_filtering(args.camera, args.calibration)
        else:
            print('Skipping real-time demo (no calibration file provided)')
if __name__ == '__main__':
    main()