import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from calibration.camera_calibrator import CameraCalibrator
from utils.image_utils import create_side_by_side_comparison

def calibrate_from_images(image_path: str, pattern_size: tuple=(9, 6), square_size: float=25.0):
    print(f'Calibrating from images in: {image_path}')
    calibrator = CameraCalibrator(pattern_size=pattern_size, square_size=square_size)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_path).glob(f'*{ext}'))
        image_files.extend(Path(image_path).glob(f'*{ext.upper()}'))
    if not image_files:
        print(f'No images found in {image_path}')
        return
    print(f'Found {len(image_files)} images')
    successful_images = 0
    for image_file in sorted(image_files):
        image = cv2.imread(str(image_file))
        if image is None:
            print(f'Failed to load {image_file}')
            continue
        if calibrator.add_calibration_image(image, visualize=False):
            successful_images += 1
    print(f'\nSuccessfully processed {successful_images}/{len(image_files)} images')
    if not calibrator.can_calibrate():
        print(f'\nInsufficient images for calibration. Need at least {calibrator._get_min_images()} images.')
        return
    print('\nPerforming calibration...')
    results = calibrator.calibrate()
    calibrator.print_camera_parameters()
    quality = calibrator.get_calibration_quality()
    print(f"\nCalibration Quality: {quality['quality']}")
    print(f"Status: {quality['status']}")
    if quality['recommendations']:
        print('\nRecommendations:')
        for rec in quality['recommendations']:
            print(f'  - {rec}')
    output_path = 'camera_calibration.pkl'
    calibrator.save_calibration(output_path)
    print('\nTesting calibration on sample image...')
    test_image = cv2.imread(str(image_files[0]))
    if test_image is not None:
        undistorted = cv2.undistort(test_image, calibrator.camera_matrix, calibrator.dist_coeffs)
        comparison = create_side_by_side_comparison(test_image, undistorted, 'Original', 'Undistorted')
        cv2.imshow('Calibration Test', comparison)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    return calibrator

def calibrate_from_live_camera(camera_id: int=0, pattern_size: tuple=(9, 6), square_size: float=25.0):
    print(f'Starting live calibration with camera {camera_id}')
    print('Instructions:')
    print('1. Hold the checkerboard at different angles and distances')
    print('2. Ensure it covers different parts of the image')
    print('3. Press SPACE to capture calibration image')
    print("4. Press 'c' to calibrate when enough images are collected")
    print("5. Press 'q' to quit")
    calibrator = CameraCalibrator(pattern_size=pattern_size, square_size=square_size)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'Failed to open camera {camera_id}')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    captured_images = 0
    last_detection_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to capture frame')
            break
        found, corners = calibrator.detect_checkerboard(frame, enhance=True)
        if found:
            frame = cv2.drawChessboardCorners(frame, pattern_size, corners, found)
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - last_detection_time > 1.0:
                status_text = f'Images: {captured_images}/{calibrator._get_min_images()} - Press SPACE to capture'
                color = (0, 255, 0) if calibrator.can_calibrate() else (0, 255, 255)
            else:
                status_text = 'Wait...'
                color = (0, 0, 255)
        else:
            status_text = f'Images: {captured_images}/{calibrator._get_min_images()} - No pattern detected'
            color = (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, 'SPACE: Capture | C: Calibrate | Q: Quit', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Live Calibration', frame)
        key = cv2.waitKey(1) & 255
        if key == ord(' '):
            if found:
                if calibrator.add_calibration_image(frame, visualize=False):
                    captured_images += 1
                    last_detection_time = cv2.getTickCount() / cv2.getTickFrequency()
                    print(f'Captured image {captured_images}')
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, 'CAPTURED!', (frame.shape[1] // 2 - 100, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.imshow('Live Calibration', frame_copy)
                    cv2.waitKey(500)
            else:
                print('No checkerboard detected - cannot capture')
        elif key == ord('c'):
            if calibrator.can_calibrate():
                print('\nPerforming calibration...')
                try:
                    results = calibrator.calibrate()
                    calibrator.print_camera_parameters()
                    calibrator.save_calibration('camera_calibration.pkl')
                    ret, test_frame = cap.read()
                    if ret:
                        undistorted = cv2.undistort(test_frame, calibrator.camera_matrix, calibrator.dist_coeffs)
                        comparison = create_side_by_side_comparison(test_frame, undistorted, 'Original', 'Undistorted')
                        cv2.imshow('Calibration Result', comparison)
                        cv2.waitKey(3000)
                    break
                except Exception as e:
                    print(f'Calibration failed: {e}')
            else:
                print(f'Need at least {calibrator._get_min_images()} images for calibration')
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return calibrator

def main():
    parser = argparse.ArgumentParser(description='Camera Calibration Script')
    parser.add_argument('--mode', choices=['images', 'live'], required=True, help='Calibration mode: images from directory or live camera')
    parser.add_argument('--path', type=str, help='Path to calibration images directory')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for live mode')
    parser.add_argument('--pattern_width', type=int, default=9, help='Checkerboard width (corners)')
    parser.add_argument('--pattern_height', type=int, default=6, help='Checkerboard height (corners)')
    parser.add_argument('--square_size', type=float, default=25.0, help='Square size in millimeters')
    args = parser.parse_args()
    pattern_size = (args.pattern_width, args.pattern_height)
    if args.mode == 'images':
        if not args.path:
            print('Error: --path required for images mode')
            return
        if not os.path.exists(args.path):
            print(f'Error: Path {args.path} does not exist')
            return
        calibrator = calibrate_from_images(args.path, pattern_size, args.square_size)
    elif args.mode == 'live':
        calibrator = calibrate_from_live_camera(args.camera, pattern_size, args.square_size)
    print('\nCalibration complete!')
if __name__ == '__main__':
    main()