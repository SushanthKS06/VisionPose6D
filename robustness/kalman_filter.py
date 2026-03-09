import numpy as np
from typing import Tuple, Optional, Dict, Any
import cv2
from ..utils.math_utils import rodrigues_to_matrix, matrix_to_rodrigues

class PoseKalmanFilter:

    def __init__(self, process_noise_position: float=1.0, process_noise_rotation: float=0.1, measurement_noise_position: float=5.0, measurement_noise_rotation: float=0.5, initial_uncertainty_position: float=100.0, initial_uncertainty_rotation: float=10.0):
        self.state_dim = 12
        self.measurement_dim = 6
        self.kf = cv2.KalmanFilter(self.state_dim, self.measurement_dim)
        self.kf.transitionMatrix = np.eye(self.state_dim, dtype=np.float32)
        dt = 1.0
        self.kf.transitionMatrix[0:3, 3:6] = np.eye(3) * dt
        self.kf.transitionMatrix[6:9, 9:12] = np.eye(3) * dt
        self.kf.measurementMatrix = np.zeros((self.measurement_dim, self.state_dim), dtype=np.float32)
        self.kf.measurementMatrix[0:3, 0:3] = np.eye(3)
        self.kf.measurementMatrix[3:6, 6:9] = np.eye(3)
        q_pos = process_noise_position
        q_rot = process_noise_rotation
        self.kf.processNoiseCov = np.eye(self.state_dim, dtype=np.float32)
        self.kf.processNoiseCov[0:3, 0:3] *= q_pos * q_pos
        self.kf.processNoiseCov[3:6, 3:6] *= q_pos * q_pos
        self.kf.processNoiseCov[6:9, 6:9] *= q_rot * q_rot
        self.kf.processNoiseCov[9:12, 9:12] *= q_rot * q_rot
        r_pos = measurement_noise_position
        r_rot = measurement_noise_rotation
        self.kf.measurementNoiseCov = np.eye(self.measurement_dim, dtype=np.float32)
        self.kf.measurementNoiseCov[0:3, 0:3] *= r_pos * r_pos
        self.kf.measurementNoiseCov[3:6, 3:6] *= r_rot * r_rot
        p_pos = initial_uncertainty_position
        p_rot = initial_uncertainty_rotation
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32)
        self.kf.errorCovPost[0:3, 0:3] *= p_pos * p_pos
        self.kf.errorCovPost[3:6, 3:6] *= p_pos * p_pos
        self.kf.errorCovPost[6:9, 6:9] *= p_rot * p_rot
        self.kf.errorCovPost[9:12, 9:12] *= p_rot * p_rot
        self.kf.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)
        self.initialized = False
        self.measurement_count = 0
        self.prediction_count = 0
        self.last_measurement_time = 0
        self.adaptive_noise_enabled = True
        self.innovation_threshold = 5.0

    def initialize(self, position: np.ndarray, rotation: np.ndarray):
        self.kf.statePost[0:3] = position
        self.kf.statePost[3:6] = 0
        self.kf.statePost[6:9] = rotation
        self.kf.statePost[9:12] = 0
        self.initialized = True
        self.measurement_count = 1
        print('Kalman filter initialized')

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            return (np.zeros((3, 1)), np.zeros((3, 1)))
        predicted_state = self.kf.predict()
        predicted_position = predicted_state[0:3]
        predicted_rotation = predicted_state[6:9]
        self.prediction_count += 1
        return (predicted_position, predicted_rotation)

    def update(self, position: np.ndarray, rotation: np.ndarray, confidence: float=1.0) -> Tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            self.initialize(position, rotation)
            return (position, rotation)
        measurement = np.vstack([position, rotation])
        if self.adaptive_noise_enabled:
            self._update_measurement_noise(confidence)
        self.kf.correct(measurement)
        filtered_position = self.kf.statePost[0:3]
        filtered_rotation = self.kf.statePost[6:9]
        self.measurement_count += 1
        return (filtered_position, filtered_rotation)

    def _update_measurement_noise(self, confidence: float):
        noise_multiplier = 1.0 / (confidence + 0.1)
        base_noise = self.kf.measurementNoiseCov.copy()
        self.kf.measurementNoiseCov = base_noise * noise_multiplier

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            return (np.zeros((3, 1)), np.zeros((3, 1)))
        linear_velocity = self.kf.statePost[3:6]
        angular_velocity = self.kf.statePost[9:12]
        return (linear_velocity, angular_velocity)

    def get_uncertainty(self) -> Dict[str, np.ndarray]:
        if not self.initialized:
            return {'position_uncertainty': np.full(3, np.inf), 'rotation_uncertainty': np.full(3, np.inf)}
        error_cov = self.kf.errorCovPost
        position_uncertainty = np.sqrt(np.diag(error_cov[0:3, 0:3]))
        rotation_uncertainty = np.sqrt(np.diag(error_cov[6:9, 6:9]))
        return {'position_uncertainty': position_uncertainty, 'rotation_uncertainty': rotation_uncertainty}

    def detect_outlier(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        if not self.initialized:
            return False
        predicted_pos, predicted_rot = self.predict()
        innovation_pos = position - predicted_pos
        innovation_rot = rotation - predicted_rot
        uncertainty = self.get_uncertainty()
        pos_nis = np.sum((innovation_pos.flatten() / uncertainty['position_uncertainty']) ** 2)
        rot_nis = np.sum((innovation_rot.flatten() / uncertainty['rotation_uncertainty']) ** 2)
        return pos_nis > self.innovation_threshold or rot_nis > self.innovation_threshold

    def reset(self):
        self.kf.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32)
        self.initialized = False
        self.measurement_count = 0
        self.prediction_count = 0
        print('Kalman filter reset')

    def get_statistics(self) -> Dict[str, Any]:
        stats = {'initialized': self.initialized, 'measurement_count': self.measurement_count, 'prediction_count': self.prediction_count, 'update_rate': self.measurement_count / (self.measurement_count + self.prediction_count) if self.measurement_count + self.prediction_count > 0 else 0}
        if self.initialized:
            velocity_pos, velocity_rot = self.get_velocity()
            uncertainty = self.get_uncertainty()
            stats.update({'linear_velocity_magnitude': np.linalg.norm(velocity_pos), 'angular_velocity_magnitude': np.linalg.norm(velocity_rot), 'position_uncertainty_magnitude': np.linalg.norm(uncertainty['position_uncertainty']), 'rotation_uncertainty_magnitude': np.linalg.norm(uncertainty['rotation_uncertainty'])})
        return stats

class AdaptivePoseFilter:

    def __init__(self):
        self.kalman_filter = PoseKalmanFilter()
        self.outlier_threshold = 3.0
        self.max_consecutive_outliers = 5
        self.consecutive_outliers = 0
        self.last_valid_pose = None
        self.pose_history = []
        self.max_history_size = 30
        self.total_measurements = 0
        self.rejected_measurements = 0
        self.predicted_poses = 0

    def process_measurement(self, position: np.ndarray, rotation: np.ndarray, confidence: float=1.0, timestamp: Optional[float]=None) -> Dict[str, Any]:
        self.total_measurements += 1
        result = {'filtered_position': position.copy(), 'filtered_rotation': rotation.copy(), 'is_outlier': False, 'is_predicted': False, 'confidence': confidence, 'uncertainty': {'position_uncertainty': np.full(3, np.inf), 'rotation_uncertainty': np.full(3, np.inf)}}
        if not self.kalman_filter.initialized:
            self.kalman_filter.initialize(position, rotation)
            self.last_valid_pose = (position.copy(), rotation.copy())
            return result
        is_outlier = self.kalman_filter.detect_outlier(position, rotation)
        if is_outlier and confidence < 0.5:
            self.consecutive_outliers += 1
            self.rejected_measurements += 1
            if self.consecutive_outliers <= self.max_consecutive_outliers:
                pred_pos, pred_rot = self.kalman_filter.predict()
                result['filtered_position'] = pred_pos
                result['filtered_rotation'] = pred_rot
                result['is_outlier'] = True
                result['is_predicted'] = True
                result['confidence'] = 0.3
                self.predicted_poses += 1
            else:
                print('Too many outliers, resetting filter')
                self.kalman_filter.reset()
                self.kalman_filter.initialize(position, rotation)
                self.consecutive_outliers = 0
        else:
            filtered_pos, filtered_rot = self.kalman_filter.update(position, rotation, confidence)
            result['filtered_position'] = filtered_pos
            result['filtered_rotation'] = filtered_rot
            result['is_outlier'] = False
            result['is_predicted'] = False
            uncertainty = self.kalman_filter.get_uncertainty()
            result['uncertainty'] = uncertainty
            self.last_valid_pose = (filtered_pos.copy(), filtered_rot.copy())
            self.consecutive_outliers = 0
        self.pose_history.append({'timestamp': timestamp or 0, 'position': result['filtered_position'].copy(), 'rotation': result['filtered_rotation'].copy(), 'confidence': result['confidence'], 'is_outlier': result['is_outlier'], 'is_predicted': result['is_predicted']})
        if len(self.pose_history) > self.max_history_size:
            self.pose_history.pop(0)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        kf_stats = self.kalman_filter.get_statistics()
        stats = {'total_measurements': self.total_measurements, 'rejected_measurements': self.rejected_measurements, 'predicted_poses': self.predicted_poses, 'rejection_rate': self.rejected_measurements / self.total_measurements if self.total_measurements > 0 else 0, 'prediction_rate': self.predicted_poses / self.total_measurements if self.total_measurements > 0 else 0, 'consecutive_outliers': self.consecutive_outliers, 'history_size': len(self.pose_history)}
        stats.update(kf_stats)
        return stats

    def reset(self):
        self.kalman_filter.reset()
        self.consecutive_outliers = 0
        self.last_valid_pose = None
        self.pose_history.clear()
        self.total_measurements = 0
        self.rejected_measurements = 0
        self.predicted_poses = 0
        print('Adaptive pose filter reset')