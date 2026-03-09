import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pose_estimation.pose_estimator import PoseEstimator, PoseResult
from aruco_tracking.aruco_detector import ArUcoDetector
from calibration.camera_calibrator import CameraCalibrator
from utils.math_utils import homogeneous_transform

def create_synthetic_test_data(num_points: int=8, noise_level: float=0.5) -> tuple:
    np.random.seed(42)
    cube_size = 50.0
    object_points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                object_points.append([x * cube_size / 2, y * cube_size / 2, z * cube_size / 2])
    object_points = np.array(object_points[:num_points], dtype=np.float32)
    true_rotation = np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]])
    true_translation = np.array([[100], [50], [300]])
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((1, 5), dtype=np.float32)
    true_rvec = cv2.Rodrigues(true_rotation)[0]
    projected_points, _ = cv2.projectPoints(object_points, true_rvec, true_translation, K, dist_coeffs)
    noise = np.random.normal(0, noise_level, projected_points.shape)
    image_points = projected_points + noise
    return (object_points, image_points.reshape(-1, 2), {'rotation': true_rotation, 'translation': true_translation, 'camera_matrix': K, 'dist_coeffs': dist_coeffs})

def test_pose_algorithms(estimator: PoseEstimator, object_points: np.ndarray, image_points: np.ndarray) -> dict:
    algorithms = ['SOLVEPNP_EPNP', 'SOLVEPNP_P3P', 'SOLVEPNP_ITERATIVE', 'SOLVEPNP_DLS', 'SOLVEPNP_UPNP']
    results = {}
    print('Testing PnP algorithms...')
    print('-' * 60)
    for algorithm in algorithms:
        try:
            result = estimator.estimate_pose(object_points, image_points, algorithm=algorithm, use_ransac=False, refine=True)
            results[algorithm] = result
            print(f'{algorithm:20s}: Error={result.reprojection_error:6.2f}px, Conf={result.confidence:5.2f}, Time={result.processing_time * 1000:6.1f}ms')
        except Exception as e:
            print(f'{algorithm:20s}: FAILED - {e}')
    try:
        ransac_result = estimator.estimate_pose(object_points, image_points, use_ransac=True, refine=True)
        results['RANSAC'] = ransac_result
        print(f"{'RANSAC':20s}: Error={ransac_result.reprojection_error:6.2f}px, Conf={ransac_result.confidence:5.2f}, Inliers={ransac_result.num_inliers:3d}, Time={ransac_result.processing_time * 1000:6.1f}ms")
    except Exception as e:
        print(f"{'RANSAC':20s}: FAILED - {e}")
    print('-' * 60)
    return results

def analyze_pose_quality(results: dict, true_pose: dict=None) -> None:
    print('\nPose Quality Analysis')
    print('=' * 60)
    sorted_results = sorted(results.items(), key=lambda x: x[1].reprojection_error)
    print('Ranking by Reprojection Error:')
    for i, (algorithm, result) in enumerate(sorted_results, 1):
        print(f'{i:2d}. {algorithm:20s}: {result.reprojection_error:6.3f}px')
    print(f'\nBest algorithm: {sorted_results[0][0]}')
    best_result = sorted_results[0][1]
    print(f'\nBest Pose Details:')
    print(f'  Position (mm): [{best_result.position[0]:7.2f}, {best_result.position[1]:7.2f}, {best_result.position[2]:7.2f}]')
    print(f'  Distance (mm): {best_result.distance:7.2f}')
    print(f'  Euler angles (deg): [{best_result.euler_angles[0]:6.1f}, {best_result.euler_angles[1]:6.1f}, {best_result.euler_angles[2]:6.1f}]')
    print(f'  Reprojection error: {best_result.reprojection_error:6.3f} pixels')
    print(f'  Confidence: {best_result.confidence:6.3f}')
    if true_pose:
        print(f'\nGround Truth Comparison:')
        print(f"  True Position: [{true_pose['translation'][0][0]:7.2f}, {true_pose['translation'][1][0]:7.2f}, {true_pose['translation'][2][0]:7.2f}]")
        pos_error = np.linalg.norm(best_result.position - true_pose['translation'].flatten())
        print(f'  Position Error: {pos_error:7.2f} mm')
        true_rvec = cv2.Rodrigues(true_pose['rotation'])[0]
        rot_error = cv2.norm(best_result.rotation_vector, true_rvec)
        print(f'  Rotation Error: {np.degrees(rot_error):7.2f} degrees')

def demonstrate_temporal_filtering(estimator: PoseEstimator, object_points: np.ndarray, image_points: np.ndarray, num_frames: int=20) -> None:
    print(f'\nTemporal Filtering Demonstration ({num_frames} frames)')
    print('=' * 60)
    estimator.reset_history()
    raw_poses = []
    filtered_poses = []
    for frame in range(num_frames):
        noise = np.random.normal(0, 0.3, image_points.shape)
        noisy_image_points = image_points + noise
        raw_result = estimator.estimate_pose(object_points, noisy_image_points)
        raw_poses.append(raw_result)
        if frame > 0:
            filtered_result = estimator.filter_pose_temporal(raw_result, alpha=0.3)
        else:
            filtered_result = raw_result
        filtered_poses.append(filtered_result)
        if frame % 5 == 0 or frame < 5:
            print(f'Frame {frame:2d}: Raw Error={raw_result.reprojection_error:5.2f}px, Filtered Error={filtered_result.reprojection_error:5.2f}px')
    raw_errors = [p.reprojection_error for p in raw_poses if p.success]
    filtered_errors = [p.reprojection_error for p in filtered_poses if p.success]
    print(f'\nFiltering Statistics:')
    print(f'  Raw poses - Mean error: {np.mean(raw_errors):5.2f}px, Std: {np.std(raw_errors):5.2f}px')
    print(f'  Filtered - Mean error: {np.mean(filtered_errors):5.2f}px, Std: {np.std(filtered_errors):5.2f}px')
    print(f'  Error reduction: {(np.std(raw_errors) - np.std(filtered_errors)) / np.std(raw_errors) * 100:5.1f}%')

def test_with_real_image(estimator: PoseEstimator, detector: ArUcoDetector, image_path: str) -> None:
    print(f'\nReal Image Pose Estimation: {image_path}')
    print('=' * 60)
    image = cv2.imread(image_path)
    if image is None:
        print(f'Failed to load image: {image_path}')
        return
    detection_results = detector.detect_markers(image, estimator.camera_matrix, estimator.dist_coeffs)
    if detection_results['num_markers'] == 0:
        print('No markers detected in image')
        return
    for detection in detection_results['detections']:
        marker_id = detection['id']
        print(f'\nMarker {marker_id}:')
        if 'pose_valid' in detection and detection['pose_valid']:
            object_points = detector.marker_object_points
            image_points = detection['corners']
            multi_results = estimator.multi_hypothesis_pose_estimation(object_points, image_points)
            if multi_results:
                best_result = multi_results[0]
                print(f'  Best Algorithm: {best_result.algorithm}')
                print(f'  Position (mm): [{best_result.position[0]:6.1f}, {best_result.position[1]:6.1f}, {best_result.position[2]:6.1f}]')
                print(f'  Distance (mm): {best_result.distance:6.1f}')
                print(f'  Euler angles (deg): [{best_result.euler_angles[0]:5.1f}, {best_result.euler_angles[1]:5.1f}, {best_result.euler_angles[2]:5.1f}]')
                print(f'  Reprojection Error: {best_result.reprojection_error:5.2f} pixels')
                print(f'  Confidence: {best_result.confidence:5.2f}')
                uncertainty = estimator.estimate_pose_uncertainty(best_result)
                print(f"  Position Uncertainty (mm): [{uncertainty['position_uncertainty'][0]:4.1f}, {uncertainty['position_uncertainty'][1]:4.1f}, {uncertainty['position_uncertainty'][2]:4.1f}]")
                print(f"  Orientation Uncertainty (deg): [{uncertainty['orientation_uncertainty'][0]:4.1f}, {uncertainty['orientation_uncertainty'][1]:4.1f}, {uncertainty['orientation_uncertainty'][2]:4.1f}]")
                is_valid = estimator.validate_pose(best_result)
                print(f"  Valid: {('Yes' if is_valid else 'No')}")
            else:
                print('  All algorithms failed')
        else:
            print('  Pose estimation failed')

def live_pose_estimation_demo(estimator: PoseEstimator, detector: ArUcoDetector, camera_id: int=0) -> None:
    print(f'\nLive Pose Estimation Demo (Camera {camera_id})')
    print('=' * 60)
    print('Controls: SPACE=Save frame, Q=Quit')
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'Failed to open camera {camera_id}')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detection_results = detector.detect_markers(frame, estimator.camera_matrix, estimator.dist_coeffs)
        for detection in detection_results['detections']:
            if 'pose_valid' in detection and detection['pose_valid']:
                marker_id = detection['id']
                pose_text = f"ID:{marker_id} D:{detection['distance']:.0f}mm"
                center = detection['center'].astype(int)
                cv2.putText(frame, pose_text, tuple(center + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        stats = estimator.get_pose_statistics()
        stats_text = f"Frames: {frame_count} | Success Rate: {stats['success_rate'] * 100:.1f}%"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Live Pose Estimation', frame)
        key = cv2.waitKey(1) & 255
        if key == ord(' '):
            timestamp = cv2.getTickCount()
            output_file = f'pose_estimation_{timestamp}.jpg'
            cv2.imwrite(output_file, frame)
            print(f'Frame saved: {output_file}')
        elif key == ord('q'):
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f'\nFinal Statistics:')
    print(f'Total frames: {frame_count}')
    print(f"Success rate: {stats['success_rate'] * 100:.1f}%")
    print(f"Mean error: {stats['mean_error']:.2f} pixels")

def load_calibration(calibration_path: str) -> tuple:
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f'Calibration file not found: {calibration_path}')
    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(calibration_path):
        raise ValueError('Failed to load calibration')
    return (calibrator.camera_matrix, calibrator.dist_coeffs)

def main():
    parser = argparse.ArgumentParser(description='6-DoF Pose Estimation Demo')
    parser.add_argument('--calibration', required=True, help='Camera calibration file')
    parser.add_argument('--synthetic', action='store_true', help='Test with synthetic data')
    parser.add_argument('--image', type=str, help='Test with real image')
    parser.add_argument('--camera', type=int, help='Live camera demo')
    parser.add_argument('--points', type=int, default=8, help='Number of synthetic points')
    parser.add_argument('--noise', type=float, default=0.5, help='Noise level for synthetic data')
    parser.add_argument('--frames', type=int, default=20, help='Frames for temporal demo')
    args = parser.parse_args()
    try:
        camera_matrix, dist_coeffs = load_calibration(args.calibration)
    except Exception as e:
        print(f'Error loading calibration: {e}')
        return
    estimator = PoseEstimator(camera_matrix, dist_coeffs)
    detector = ArUcoDetector()
    print('6-DoF Pose Estimation Demonstration')
    print('=' * 60)
    print(f'Camera Matrix:\n{camera_matrix}')
    print(f'Distortion Coefficients: {dist_coeffs[0]}')
    if args.synthetic:
        print(f'\nGenerating synthetic test data ({args.points} points, noise={args.noise})...')
        object_points, image_points, true_pose = create_synthetic_test_data(args.points, args.noise)
        results = test_pose_algorithms(estimator, object_points, image_points)
        analyze_pose_quality(results, true_pose)
        demonstrate_temporal_filtering(estimator, object_points, image_points, args.frames)
    elif args.image:
        if not os.path.exists(args.image):
            print(f'Image not found: {args.image}')
            return
        test_with_real_image(estimator, detector, args.image)
    elif args.camera is not None:
        live_pose_estimation_demo(estimator, detector, args.camera)
    else:
        print('Error: Must specify one of --synthetic, --image, or --camera')
        print('Use --help for usage information')
if __name__ == '__main__':
    main()