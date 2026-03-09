import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from ..utils.math_utils import compute_reprojection_error
from ..utils.image_utils import draw_checkerboard_corners, enhance_contrast

class CameraCalibrator:

    def __init__(self, pattern_size: Tuple[int, int]=(9, 6), square_size: float=25.0, criteria: Tuple=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.criteria = criteria
        self.object_points = []
        self.image_points = []
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None
        self.calibrated = False
        self._prepare_object_points()

    def _prepare_object_points(self):
        pattern_width, pattern_height = self.pattern_size
        objp = np.zeros((pattern_width * pattern_height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_width, 0:pattern_height].T.reshape(-1, 2)
        objp *= self.square_size
        self.objp_template = objp

    def detect_checkerboard(self, image: np.ndarray, enhance: bool=True) -> Tuple[bool, Optional[np.ndarray]]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if enhance:
            gray = enhance_contrast(gray, method='clahe')
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            if self._validate_corners(corners):
                return (True, corners)
            else:
                return (False, None)
        return (False, None)

    def _validate_corners(self, corners: np.ndarray) -> bool:
        if np.any(~np.isfinite(corners)):
            return False
        expected_corners = self.pattern_size[0] * self.pattern_size[1]
        if len(corners) != expected_corners:
            return False
        return True

    def add_calibration_image(self, image: np.ndarray, visualize: bool=False) -> bool:
        if self.image_size is None:
            self.image_size = (image.shape[1], image.shape[0])
        found, corners = self.detect_checkerboard(image)
        if found:
            self.object_points.append(self.objp_template.copy())
            self.image_points.append(corners)
            if visualize:
                result = draw_checkerboard_corners(image, corners, self.pattern_size, True)
                cv2.imshow('Calibration Detection', result)
                cv2.waitKey(1000)
            print(f'Added calibration image {len(self.object_points)}/{self._get_min_images()}')
            return True
        else:
            if visualize:
                result = draw_checkerboard_corners(image, None, self.pattern_size, False)
                cv2.imshow('Calibration Detection', result)
                cv2.waitKey(500)
            print('Checkerboard not detected in image')
            return False

    def _get_min_images(self) -> int:
        return 10

    def can_calibrate(self) -> bool:
        return len(self.object_points) >= self._get_min_images()

    def calibrate(self, flags: int=cv2.CALIB_FIX_ASPECT_RATIO) -> Dict[str, Any]:
        if not self.can_calibrate():
            raise ValueError(f'Need at least {self._get_min_images()} calibration images')
        print(f'Calibrating with {len(self.object_points)} images...')
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, self.image_size, None, None, flags=flags)
        if not ret:
            raise RuntimeError('Camera calibration failed')
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.calibrated = True
        errors = []
        for i in range(len(self.object_points)):
            error = compute_reprojection_error(self.object_points[i], self.image_points[i], self.rvecs[i], self.tvecs[i], camera_matrix, dist_coeffs)
            errors.append(error)
        self.reprojection_error = {'mean': np.mean(errors), 'std': np.std(errors), 'max': np.max(errors), 'min': np.min(errors), 'individual': errors}
        results = {'success': True, 'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs, 'reprojection_error': self.reprojection_error, 'num_images': len(self.object_points), 'image_size': self.image_size, 'pattern_size': self.pattern_size, 'square_size': self.square_size}
        print(f'Calibration complete!')
        print(f"Mean reprojection error: {self.reprojection_error['mean']:.3f} pixels")
        print(f"Std deviation: {self.reprojection_error['std']:.3f} pixels")
        return results

    def get_calibration_quality(self) -> Dict[str, str]:
        if not self.calibrated:
            return {'status': 'Not calibrated'}
        mean_error = self.reprojection_error['mean']
        std_error = self.reprojection_error['std']
        assessment = {}
        if mean_error < 0.3:
            assessment['quality'] = 'Excellent'
            assessment['status'] = 'Production ready'
        elif mean_error < 0.5:
            assessment['quality'] = 'Good'
            assessment['status'] = 'Acceptable for most applications'
        elif mean_error < 1.0:
            assessment['quality'] = 'Fair'
            assessment['status'] = 'May need improvement for precision applications'
        else:
            assessment['quality'] = 'Poor'
            assessment['status'] = 'Recalibration recommended'
        recommendations = []
        if mean_error > 0.5:
            recommendations.append('Collect more calibration images')
            recommendations.append('Ensure checkerboard covers entire field of view')
            recommendations.append('Use better lighting conditions')
        if std_error > 0.3:
            recommendations.append('Check for inconsistent camera focus')
            recommendations.append('Ensure checkerboard is flat during capture')
        if len(self.object_points) < 20:
            recommendations.append('Collect more images for better coverage')
        assessment['recommendations'] = recommendations
        return assessment

    def save_calibration(self, filepath: str) -> None:
        if not self.calibrated:
            raise ValueError('No calibration data to save')
        calibration_data = {'camera_matrix': self.camera_matrix, 'dist_coeffs': self.dist_coeffs, 'image_size': self.image_size, 'pattern_size': self.pattern_size, 'square_size': self.square_size, 'reprojection_error': self.reprojection_error, 'num_images': len(self.object_points), 'metadata': {'opencv_version': cv2.__version__, 'calibration_date': cv2.getTickCount()}}
        with open(filepath, 'wb') as f:
            pickle.dump(calibration_data, f)
        print(f'Calibration saved to {filepath}')

    def load_calibration(self, filepath: str) -> bool:
        try:
            with open(filepath, 'rb') as f:
                calibration_data = pickle.load(f)
            self.camera_matrix = calibration_data['camera_matrix']
            self.dist_coeffs = calibration_data['dist_coeffs']
            self.image_size = calibration_data['image_size']
            self.pattern_size = calibration_data['pattern_size']
            self.square_size = calibration_data['square_size']
            self.reprojection_error = calibration_data.get('reprojection_error')
            self.calibrated = True
            print(f'Calibration loaded from {filepath}')
            print(f"Reprojection error: {self.reprojection_error['mean']:.3f} pixels")
            return True
        except Exception as e:
            print(f'Failed to load calibration: {e}')
            return False

    def clear_calibration_data(self):
        self.object_points.clear()
        self.image_points.clear()
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None
        self.calibrated = False
        print('Calibration data cleared')

    def print_camera_parameters(self):
        if not self.calibrated:
            print('No calibration available')
            return
        print('\n' + '=' * 50)
        print('CAMERA CALIBRATION PARAMETERS')
        print('=' * 50)
        print(f'\nImage Size: {self.image_size}')
        print(f'Pattern Size: {self.pattern_size}')
        print(f'Square Size: {self.square_size} mm')
        print(f'Number of Images: {len(self.object_points)}')
        print(f'\nCamera Matrix:')
        print(self.camera_matrix)
        print(f'\nFocal Lengths: fx={self.camera_matrix[0, 0]:.2f}, fy={self.camera_matrix[1, 1]:.2f}')
        print(f'Principal Point: ({self.camera_matrix[0, 2]:.2f}, {self.camera_matrix[1, 2]:.2f})')
        print(f'\nDistortion Coefficients:')
        print(f'k1={self.dist_coeffs[0, 0]:.6f}, k2={self.dist_coeffs[0, 1]:.6f}')
        print(f'p1={self.dist_coeffs[0, 2]:.6f}, p2={self.dist_coeffs[0, 3]:.6f}')
        print(f'k3={self.dist_coeffs[0, 4]:.6f}')
        print(f"\nReprojection Error: {self.reprojection_error['mean']:.3f} ± {self.reprojection_error['std']:.3f} pixels")
        print('=' * 50)