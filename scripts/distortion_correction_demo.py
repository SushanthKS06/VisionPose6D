import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.image_utils import create_side_by_side_comparison, undistort_image
from calibration.camera_calibrator import CameraCalibrator

class DistortionCorrector:

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.distortion_maps = {}

    def correct_image(self, image: np.ndarray, use_optimized: bool=True) -> np.ndarray:
        if use_optimized:
            image_size = (image.shape[1], image.shape[0])
            if image_size not in self.distortion_maps:
                self.distortion_maps[image_size] = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, self.camera_matrix, image_size, cv2.CV_16SC2)
            map1, map2 = self.distortion_maps[image_size]
            return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        else:
            return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def analyze_distortion(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        grid_points = []
        for y in np.linspace(0, h - 1, 10):
            for x in np.linspace(0, w - 1, 10):
                grid_points.append([x, y])
        grid_points = np.array(grid_points, dtype=np.float32)
        undistorted_points = cv2.undistortPoints(grid_points.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs).reshape(-1, 2)
        undistorted_points = undistorted_points * np.array([[self.camera_matrix[0, 0], self.camera_matrix[1, 1]]]) + np.array([self.camera_matrix[0, 2], self.camera_matrix[1, 2]])
        displacements = undistorted_points - grid_points
        displacement_magnitudes = np.sqrt(np.sum(displacements ** 2, axis=1))
        analysis = {'max_displacement': np.max(displacement_magnitudes), 'mean_displacement': np.mean(displacement_magnitudes), 'center_displacement': displacement_magnitudes[len(displacement_magnitudes) // 2], 'corner_displacement': np.mean([displacement_magnitudes[0], displacement_magnitudes[9], displacement_magnitudes[-10], displacement_magnitudes[-1]]), 'distortion_type': self._classify_distortion(self.dist_coeffs)}
        return analysis

    def _classify_distortion(self, dist_coeffs: np.ndarray) -> str:
        k1, k2, p1, p2, k3 = dist_coeffs[0]
        radial_magnitude = abs(k1) + abs(k2) + abs(k3)
        tangential_magnitude = abs(p1) + abs(p2)
        if radial_magnitude < 0.01 and tangential_magnitude < 0.001:
            return 'Minimal distortion'
        elif radial_magnitude < 0.1:
            if tangential_magnitude < 0.01:
                return 'Low radial distortion'
            else:
                return 'Low radial + tangential distortion'
        elif radial_magnitude < 0.3:
            if tangential_magnitude < 0.01:
                return 'Moderate radial distortion'
            else:
                return 'Moderate radial + tangential distortion'
        else:
            return 'High distortion (wide-angle lens)'

    def create_distortion_visualization(self, image: np.ndarray) -> np.ndarray:
        undistorted = self.correct_image(image)
        comparison = create_side_by_side_comparison(image, undistorted, 'Distorted', 'Corrected')
        analysis = self.analyze_distortion(image)
        h, w = comparison.shape[:2]
        y_offset = 40
        text_bg = np.zeros((150, w, 3), dtype=np.uint8)
        texts = [f'Distortion Analysis:', f"Type: {analysis['distortion_type']}", f"Max displacement: {analysis['max_displacement']:.2f} pixels", f"Mean displacement: {analysis['mean_displacement']:.2f} pixels", f"Corner displacement: {analysis['corner_displacement']:.2f} pixels"]
        for i, text in enumerate(texts):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(text_bg, text, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        result = np.vstack([comparison, text_bg])
        return result

    def create_grid_overlay(self, image: np.ndarray, grid_size: int=50) -> np.ndarray:
        h, w = image.shape[:2]
        distorted_with_grid = image.copy()
        for x in range(0, w, grid_size):
            cv2.line(distorted_with_grid, (x, 0), (x, h), (0, 255, 0), 1)
        for y in range(0, h, grid_size):
            cv2.line(distorted_with_grid, (0, y), (w, y), (0, 255, 0), 1)
        undistorted_with_grid = self.correct_image(distorted_with_grid)
        result = create_side_by_side_comparison(distorted_with_grid, undistorted_with_grid, 'Distorted Grid', 'Corrected Grid')
        return result

def load_calibration(calibration_path: str) -> tuple:
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f'Calibration file not found: {calibration_path}')
    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(calibration_path):
        raise ValueError('Failed to load calibration')
    return (calibrator.camera_matrix, calibrator.dist_coeffs)

def main():
    parser = argparse.ArgumentParser(description='Distortion Correction Demonstration')
    parser.add_argument('--calibration', required=True, help='Path to camera calibration file (.pkl)')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--camera', type=int, help='Camera ID for live demo')
    parser.add_argument('--output', type=str, default='distortion_demo.jpg', help='Output image file')
    args = parser.parse_args()
    try:
        camera_matrix, dist_coeffs = load_calibration(args.calibration)
    except Exception as e:
        print(f'Error loading calibration: {e}')
        return
    corrector = DistortionCorrector(camera_matrix, dist_coeffs)
    print('Distortion Correction Demonstration')
    print('=' * 50)
    print(f'Camera Matrix:\n{camera_matrix}')
    print(f'\nDistortion Coefficients: {dist_coeffs[0]}')
    distortion_type = corrector._classify_distortion(dist_coeffs)
    print(f'\nDistortion Type: {distortion_type}')
    if args.image:
        if not os.path.exists(args.image):
            print(f'Image not found: {args.image}')
            return
        print(f'\nProcessing image: {args.image}')
        image = cv2.imread(args.image)
        if image is None:
            print(f'Failed to load image: {args.image}')
            return
        print('Creating distortion visualization...')
        visualization = corrector.create_distortion_visualization(image)
        cv2.imwrite(args.output, visualization)
        print(f'Visualization saved to: {args.output}')
        grid_visualization = corrector.create_grid_overlay(image)
        grid_output = args.output.replace('.jpg', '_grid.jpg')
        cv2.imwrite(grid_output, grid_visualization)
        print(f'Grid visualization saved to: {grid_output}')
        cv2.imshow('Distortion Analysis', visualization)
        cv2.imshow('Grid Distortion', grid_visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.camera is not None:
        print(f'\nStarting live camera demo with camera {args.camera}')
        print('Press SPACE to capture frame, Q to quit')
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f'Failed to open camera {args.camera}')
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            undistorted = corrector.correct_image(frame, use_optimized=True)
            comparison = create_side_by_side_comparison(frame, undistorted, 'Distorted', 'Corrected')
            cv2.imshow('Live Distortion Correction', comparison)
            key = cv2.waitKey(1) & 255
            if key == ord(' '):
                timestamp = cv2.getTickCount()
                output_file = f'distortion_demo_{timestamp}.jpg'
                cv2.imwrite(output_file, comparison)
                print(f'Frame saved to: {output_file}')
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Error: Must specify either --image or --camera')
if __name__ == '__main__':
    main()