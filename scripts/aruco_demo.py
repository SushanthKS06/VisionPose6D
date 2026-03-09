import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from aruco_tracking.aruco_detector import ArUcoDetector
from calibration.camera_calibrator import CameraCalibrator

def generate_markers(detector: ArUcoDetector, marker_ids: list, output_dir: str, marker_size_pixels: int=200):
    os.makedirs(output_dir, exist_ok=True)
    print(f'Generating {len(marker_ids)} markers...')
    for marker_id in marker_ids:
        output_path = os.path.join(output_dir, f'marker_{marker_id}.png')
        success = detector.generate_marker(marker_id, output_path, marker_size_pixels)
        if success:
            print(f' Generated marker {marker_id}')
        else:
            print(f' Failed to generate marker {marker_id}')
    print(f'Markers saved to: {output_dir}')

def detect_from_image(detector: ArUcoDetector, image_path: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, output_path: str=None):
    print(f'Detecting markers in: {image_path}')
    image = cv2.imread(image_path)
    if image is None:
        print(f'Failed to load image: {image_path}')
        return
    detection_results = detector.detect_markers(image, camera_matrix, dist_coeffs)
    print(f"Found {detection_results['num_markers']} markers")
    for detection in detection_results['detections']:
        marker_id = detection['id']
        area = detection['area']
        print(f'\nMarker {marker_id}:')
        print(f'  Area: {area:.1f} pixels²')
        if 'pose_valid' in detection and detection['pose_valid']:
            euler = detection['euler_angles']
            distance = detection['distance']
            error = detection['reprojection_error']
            print(f"  Position: ({detection['tvec'][0][0]:.1f}, {detection['tvec'][1][0]:.1f}, {detection['tvec'][2][0]:.1f}) mm")
            print(f'  Rotation: Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°')
            print(f'  Distance: {distance:.1f} mm')
            print(f'  Reprojection Error: {error:.2f} pixels')
        else:
            print('  Pose estimation failed')
    result_image = detector.draw_detections(image, detection_results, show_ids=True, show_axes=True, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    h, w = result_image.shape[:2]
    summary_text = f"Markers: {detection_results['num_markers']}"
    cv2.putText(result_image, summary_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f'Result saved to: {output_path}')
    cv2.imshow('ArUco Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_from_camera(detector: ArUcoDetector, camera_id: int, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    print(f'Starting live ArUco detection with camera {camera_id}')
    print('Controls:')
    print('  SPACE: Save current frame')
    print('  S: Save tracking data')
    print('  C: Clear tracking data')
    print('  Q: Quit')
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'Failed to open camera {camera_id}')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detection_results = detector.detect_markers(frame, camera_matrix, dist_coeffs)
        result_frame = detector.draw_detections(frame, detection_results, show_ids=True, show_axes=True, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        stable_markers = detector.get_stable_markers(min_consecutive_detections=3)
        status_lines = [f'Frame: {frame_count}', f"Markers: {detection_results['num_markers']}", f'Stable: {len(stable_markers)}', f'Tracked: {len(detector.tracked_markers)}']
        y_offset = 30
        for line in status_lines:
            cv2.putText(result_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25
        cv2.putText(result_frame, 'SPACE: Save | S: Tracking | C: Clear | Q: Quit', (10, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Live ArUco Detection', result_frame)
        key = cv2.waitKey(1) & 255
        if key == ord(' '):
            timestamp = cv2.getTickCount()
            output_file = f'aruco_detection_{timestamp}.jpg'
            cv2.imwrite(output_file, result_frame)
            print(f'Frame saved to: {output_file}')
        elif key == ord('s'):
            tracking_file = f'aruco_tracking_{timestamp}.pkl'
            detector.save_tracking_data(tracking_file)
            print(f'Tracking data saved to: {tracking_file}')
        elif key == ord('c'):
            detector.tracked_markers.clear()
            detector.frame_count = 0
            print('Tracking data cleared')
        elif key == ord('q'):
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    print('\nTracking Summary:')
    print(f'Total frames processed: {frame_count}')
    print(f'Total unique markers detected: {len(detector.tracked_markers)}')
    for marker_id, tracking_data in detector.tracked_markers.items():
        print(f'  Marker {marker_id}:')
        print(f"    Detections: {tracking_data['detection_count']}")
        print(f"    Quality score: {tracking_data['quality_score']:.2f}")
        print(f"    Consecutive detections: {tracking_data['consecutive_detections']}")

def load_calibration(calibration_path: str) -> tuple:
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f'Calibration file not found: {calibration_path}')
    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(calibration_path):
        raise ValueError('Failed to load calibration')
    return (calibrator.camera_matrix, calibrator.dist_coeffs)

def main():
    parser = argparse.ArgumentParser(description='ArUco Marker Detection Demo')
    parser.add_argument('--generate', action='store_true', help='Generate markers')
    parser.add_argument('--detect', action='store_true', help='Detect markers in image')
    parser.add_argument('--live', action='store_true', help='Live camera detection')
    parser.add_argument('--id', type=int, help='Marker ID for generation')
    parser.add_argument('--ids', nargs='+', type=int, help='Multiple marker IDs')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--image', type=str, help='Input image for detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for live mode')
    parser.add_argument('--calibration', type=str, help='Camera calibration file')
    parser.add_argument('--dictionary', type=str, default='DICT_4X4_50', choices=['DICT_4X4_50', 'DICT_4X4_100', 'DICT_5X5_50', 'DICT_5X5_100', 'DICT_6X6_50', 'DICT_6X6_100', 'DICT_7X7_50', 'DICT_7X7_100'], help='ArUco dictionary')
    parser.add_argument('--marker_size', type=float, default=50.0, help='Physical marker size in millimeters')
    parser.add_argument('--marker_pixels', type=int, default=200, help='Marker image size in pixels')
    args = parser.parse_args()
    detector = ArUcoDetector(dictionary_name=args.dictionary, marker_size=args.marker_size)
    print(f'ArUco Detector initialized:')
    print(f'  Dictionary: {args.dictionary}')
    print(f'  Marker size: {args.marker_size} mm')
    if args.generate:
        if args.ids:
            marker_ids = args.ids
        elif args.id is not None:
            marker_ids = [args.id]
        else:
            marker_ids = list(range(10))
        output_dir = args.output or 'aruco_markers'
        generate_markers(detector, marker_ids, output_dir, args.marker_pixels)
    elif args.detect:
        if not args.image:
            print('Error: --image required for detection mode')
            return
        if not args.calibration:
            print('Error: --calibration required for pose estimation')
            return
        camera_matrix, dist_coeffs = load_calibration(args.calibration)
        output_path = args.output or 'aruco_detection_result.jpg'
        detect_from_image(detector, args.image, camera_matrix, dist_coeffs, output_path)
    elif args.live:
        if not args.calibration:
            print('Error: --calibration required for pose estimation')
            return
        camera_matrix, dist_coeffs = load_calibration(args.calibration)
        detect_from_camera(detector, args.camera, camera_matrix, dist_coeffs)
    else:
        print('Error: Must specify one of --generate, --detect, or --live')
        print('Use --help for usage information')
if __name__ == '__main__':
    main()