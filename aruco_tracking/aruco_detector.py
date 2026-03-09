import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import pickle
import os
from ..utils.math_utils import compute_reprojection_error

class ArUcoDetector:

    def __init__(self, dictionary_name: str='DICT_4X4_50', marker_size: float=50.0, detection_params: Optional[Dict]=None):
        self.dictionary_name = dictionary_name
        self.marker_size = marker_size
        self.dictionary = self._get_dictionary(dictionary_name)
        self.detector_params = self._setup_detection_params(detection_params)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        self.tracked_markers = {}
        self.frame_count = 0
        self._prepare_marker_object_points()

    def _get_dictionary(self, dictionary_name: str):
        dictionary_map = {'DICT_4X4_50': cv2.aruco.DICT_4X4_50, 'DICT_4X4_100': cv2.aruco.DICT_4X4_100, 'DICT_4X4_250': cv2.aruco.DICT_4X4_250, 'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000, 'DICT_5X5_50': cv2.aruco.DICT_5X5_50, 'DICT_5X5_100': cv2.aruco.DICT_5X5_100, 'DICT_5X5_250': cv2.aruco.DICT_5X5_250, 'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000, 'DICT_6X6_50': cv2.aruco.DICT_6X6_50, 'DICT_6X6_100': cv2.aruco.DICT_6X6_100, 'DICT_6X6_250': cv2.aruco.DICT_6X6_250, 'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000, 'DICT_7X7_50': cv2.aruco.DICT_7X7_50, 'DICT_7X7_100': cv2.aruco.DICT_7X7_100, 'DICT_7X7_250': cv2.aruco.DICT_7X7_250, 'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000, 'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL}
        if dictionary_name not in dictionary_map:
            raise ValueError(f'Unknown dictionary: {dictionary_name}')
        return cv2.aruco.getPredefinedDictionary(dictionary_map[dictionary_name])

    def _setup_detection_params(self, custom_params: Optional[Dict]=None) -> cv2.aruco.DetectorParameters:
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMinAccuracy = 0.01
        params.cornerRefinementMaxIterations = 30
        params.markerBorderBits = 1
        params.perspectiveRemovePixelPerCell = 4
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        params.perspectiveRemovePixelPerCell = 8
        params.errorCorrectionRate = 0.6
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(params, key):
                    setattr(params, key, value)
        return params

    def _prepare_marker_object_points(self):
        half_size = self.marker_size / 2.0
        self.marker_object_points = np.array([[-half_size, half_size, 0], [half_size, half_size, 0], [half_size, -half_size, 0], [-half_size, -half_size, 0]], dtype=np.float32)

    def detect_markers(self, image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Dict[str, Any]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        corners, ids, rejected = self.detector.detectMarkers(gray)
        detection_results = {'corners': corners if corners else [], 'ids': ids if ids is not None else np.array([]), 'rejected': rejected if rejected else [], 'num_markers': len(corners) if corners else 0, 'detections': []}
        if corners and ids is not None:
            for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
                detection_info = {'id': int(marker_id), 'corners': corner.reshape(-1, 2), 'center': np.mean(corner.reshape(-1, 2), axis=0), 'area': cv2.contourArea(corner.reshape(-1, 2)), 'perimeter': cv2.arcLength(corner.reshape(-1, 2), True)}
                if camera_matrix is not None and dist_coeffs is not None:
                    pose_result = self.estimate_pose(corner, camera_matrix, dist_coeffs)
                    detection_info.update(pose_result)
                detection_results['detections'].append(detection_info)
        self._update_tracking(detection_results)
        self.frame_count += 1
        return detection_results

    def estimate_pose(self, marker_corners: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, method: int=cv2.SOLVEPNP_IPPE) -> Dict[str, Any]:
        try:
            success, rvec, tvec = cv2.solvePnP(self.marker_object_points, marker_corners, camera_matrix, dist_coeffs, flags=method, useExtrinsicGuess=False)
            if not success:
                return {'pose_valid': False}
            projected_points, _ = cv2.projectPoints(self.marker_object_points, rvec, tvec, camera_matrix, dist_coeffs)
            reprojection_error = cv2.norm(marker_corners.reshape(-1, 2), projected_points.reshape(-1, 2), cv2.NORM_L2) / 4.0
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
            pose_info = {'pose_valid': True, 'rvec': rvec, 'tvec': tvec, 'rotation_matrix': rotation_matrix, 'euler_angles': euler_angles, 'reprojection_error': reprojection_error, 'distance': np.linalg.norm(tvec)}
            return pose_info
        except Exception as e:
            print(f'Pose estimation failed: {e}')
            return {'pose_valid': False}

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-06
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees([x, y, z])

    def _update_tracking(self, detection_results: Dict[str, Any]):
        current_ids = set()
        for detection in detection_results['detections']:
            marker_id = detection['id']
            current_ids.add(marker_id)
            if marker_id not in self.tracked_markers:
                self.tracked_markers[marker_id] = {'first_seen': self.frame_count, 'last_seen': self.frame_count, 'detection_count': 1, 'consecutive_detections': 1, 'last_pose': detection.get('rvec', None), 'last_translation': detection.get('tvec', None), 'quality_score': self._compute_quality_score(detection)}
            else:
                tracking_data = self.tracked_markers[marker_id]
                tracking_data['last_seen'] = self.frame_count
                tracking_data['detection_count'] += 1
                if tracking_data['last_seen'] == self.frame_count - 1:
                    tracking_data['consecutive_detections'] += 1
                else:
                    tracking_data['consecutive_detections'] = 1
                if 'pose_valid' in detection and detection['pose_valid']:
                    tracking_data['last_pose'] = detection['rvec']
                    tracking_data['last_translation'] = detection['tvec']
                tracking_data['quality_score'] = self._compute_quality_score(detection)
        for marker_id in self.tracked_markers:
            if marker_id not in current_ids:
                tracking_data = self.tracked_markers[marker_id]
                tracking_data['consecutive_detections'] = 0

    def _compute_quality_score(self, detection: Dict[str, Any]) -> float:
        score = 0.0
        area = detection.get('area', 0)
        if area > 10000:
            score += 0.3
        elif area > 5000:
            score += 0.2
        elif area > 1000:
            score += 0.1
        if 'reprojection_error' in detection:
            error = detection['reprojection_error']
            if error < 1.0:
                score += 0.4
            elif error < 2.0:
                score += 0.3
            elif error < 5.0:
                score += 0.1
        corners = detection['corners']
        if len(corners) == 4:
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            if width > 0 and height > 0:
                aspect_ratio = width / height
                if 0.8 < aspect_ratio < 1.2:
                    score += 0.3
                elif 0.6 < aspect_ratio < 1.4:
                    score += 0.2
        return min(score, 1.0)

    def get_stable_markers(self, min_consecutive_detections: int=3, min_quality_score: float=0.5) -> List[int]:
        stable_markers = []
        for marker_id, tracking_data in self.tracked_markers.items():
            if tracking_data['consecutive_detections'] >= min_consecutive_detections and tracking_data['quality_score'] >= min_quality_score:
                stable_markers.append(marker_id)
        return stable_markers

    def get_marker_info(self, marker_id: int) -> Optional[Dict]:
        return self.tracked_markers.get(marker_id)

    def draw_detections(self, image: np.ndarray, detection_results: Dict[str, Any], show_ids: bool=True, show_axes: bool=False, camera_matrix: Optional[np.ndarray]=None, dist_coeffs: Optional[np.ndarray]=None) -> np.ndarray:
        result = image.copy()
        for detection in detection_results['detections']:
            corners = detection['corners'].astype(int)
            marker_id = detection['id']
            cv2.polylines(result, [corners], True, (0, 255, 0), 2)
            if show_ids:
                center = detection['center'].astype(int)
                cv2.putText(result, str(marker_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result, str(marker_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            if show_axes and 'pose_valid' in detection and detection['pose_valid']:
                if camera_matrix is not None and dist_coeffs is not None:
                    self._draw_coordinate_axes(result, detection['rvec'], detection['tvec'], camera_matrix, dist_coeffs)
            quality_score = self.tracked_markers.get(marker_id, {}).get('quality_score', 0)
            color = (0, int(255 * quality_score), int(255 * (1 - quality_score)))
            cv2.circle(result, tuple(corners[0]), 5, color, -1)
        return result

    def _draw_coordinate_axes(self, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, axis_length: float=30.0) -> None:
        axis_points = np.array([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]], dtype=np.float32)
        projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2).astype(int)
        origin = projected_points[0]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        labels = ['X', 'Y', 'Z']
        for i, (point, color, label) in enumerate(zip(projected_points[1:], colors, labels)):
            cv2.line(image, origin, point, color, 3)
            cv2.putText(image, label, tuple(point + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def generate_marker(self, marker_id: int, output_path: str, marker_size_pixels: int=200) -> bool:
        try:
            marker_image = cv2.aruco.generateImageMarker(self.dictionary, marker_id, marker_size_pixels)
            border_size = marker_size_pixels // 10
            marker_with_border = cv2.copyMakeBorder(marker_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)
            cv2.imwrite(output_path, marker_with_border)
            print(f'Generated marker {marker_id} saved to {output_path}')
            return True
        except Exception as e:
            print(f'Failed to generate marker {marker_id}: {e}')
            return False

    def save_tracking_data(self, filepath: str) -> None:
        tracking_data = {'dictionary_name': self.dictionary_name, 'marker_size': self.marker_size, 'tracked_markers': self.tracked_markers, 'frame_count': self.frame_count}
        with open(filepath, 'wb') as f:
            pickle.dump(tracking_data, f)
        print(f'Tracking data saved to {filepath}')

    def load_tracking_data(self, filepath: str) -> bool:
        try:
            with open(filepath, 'rb') as f:
                tracking_data = pickle.load(f)
            self.tracked_markers = tracking_data['tracked_markers']
            self.frame_count = tracking_data['frame_count']
            print(f'Tracking data loaded from {filepath}')
            return True
        except Exception as e:
            print(f'Failed to load tracking data: {e}')
            return False