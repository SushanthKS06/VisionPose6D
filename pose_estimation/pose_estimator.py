import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from ..utils.math_utils import rodrigues_to_matrix, matrix_to_rodrigues, homogeneous_transform, decompose_homogeneous, compute_reprojection_error

@dataclass
class PoseResult:
    success: bool
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    rotation_vector: np.ndarray
    reprojection_error: float
    confidence: float
    algorithm: str
    num_inliers: int = 0
    processing_time: float = 0.0

    @property
    def euler_angles(self) -> Tuple[float, float, float]:
        rotation = R.from_matrix(self.rotation_matrix)
        return rotation.as_euler('xyz', degrees=True)

    @property
    def homogeneous_transform(self) -> np.ndarray:
        return homogeneous_transform(self.rotation_matrix, self.translation_vector)

    @property
    def position(self) -> np.ndarray:
        return self.translation_vector.flatten()

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.position)

class PoseEstimator:

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, default_algorithm: str='SOLVEPNP_EPNP'):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.default_algorithm = default_algorithm
        self.algorithms = {'SOLVEPNP_ITERATIVE': cv2.SOLVEPNP_ITERATIVE, 'SOLVEPNP_P3P': cv2.SOLVEPNP_P3P, 'SOLVEPNP_EPNP': cv2.SOLVEPNP_EPNP, 'SOLVEPNP_DLS': cv2.SOLVEPNP_DLS, 'SOLVEPNP_UPNP': cv2.SOLVEPNP_UPNP, 'SOLVEPNP_AP3P': cv2.SOLVEPNP_AP3P, 'SOLVEPNP_IPPE': cv2.SOLVEPNP_IPPE, 'SOLVEPNP_IPPE_SQUARE': cv2.SOLVEPNP_IPPE_SQUARE, 'SOLVEPNP_SQPNP': cv2.SOLVEPNP_SQPNP}
        self.ransac_params = {'iterationsCount': 100, 'reprojectionError': 8.0, 'confidence': 0.99, 'flags': cv2.SOLVEPNP_EPNP}
        self.pose_history = []
        self.max_history_size = 10
        self.uncertainty_params = {'position_noise_std': 1.0, 'orientation_noise_std': 1.0, 'outlier_threshold': 5.0}

    def estimate_pose(self, object_points: np.ndarray, image_points: np.ndarray, algorithm: Optional[str]=None, use_ransac: bool=False, refine: bool=True) -> PoseResult:
        import time
        start_time = time.time()
        if len(object_points) != len(image_points):
            raise ValueError('Object points and image points must have same length')
        if len(object_points) < 4:
            raise ValueError('Need at least 4 point correspondences for pose estimation')
        if algorithm is None:
            algorithm = self.default_algorithm
        if algorithm not in self.algorithms:
            raise ValueError(f'Unknown algorithm: {algorithm}')
        flags = self.algorithms[algorithm]
        try:
            if use_ransac:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, self.camera_matrix, self.dist_coeffs, **self.ransac_params)
                num_inliers = len(inliers) if inliers is not None else 0
            else:
                success, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=flags, useExtrinsicGuess=False)
                num_inliers = len(object_points)
            if not success:
                return PoseResult(success=False, rotation_matrix=np.eye(3), translation_vector=np.zeros((3, 1)), rotation_vector=np.zeros((3, 1)), reprojection_error=float('inf'), confidence=0.0, algorithm=algorithm, num_inliers=0, processing_time=time.time() - start_time)
            if refine:
                rvec, tvec = self._refine_pose(object_points, image_points, rvec, tvec)
            rotation_matrix = rodrigues_to_matrix(rvec)
            reprojection_error = self._compute_reprojection_error(object_points, image_points, rvec, tvec)
            confidence = self._compute_confidence(reprojection_error, num_inliers, len(object_points))
            result = PoseResult(success=True, rotation_matrix=rotation_matrix, translation_vector=tvec, rotation_vector=rvec, reprojection_error=reprojection_error, confidence=confidence, algorithm=algorithm, num_inliers=num_inliers, processing_time=time.time() - start_time)
            self._update_pose_history(result)
            return result
        except Exception as e:
            print(f'Pose estimation failed: {e}')
            return PoseResult(success=False, rotation_matrix=np.eye(3), translation_vector=np.zeros((3, 1)), rotation_vector=np.zeros((3, 1)), reprojection_error=float('inf'), confidence=0.0, algorithm=algorithm, num_inliers=0, processing_time=time.time() - start_time)

    def _refine_pose(self, object_points: np.ndarray, image_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, max_iterations: int=20) -> Tuple[np.ndarray, np.ndarray]:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 1e-06)
        success, rvec_refined, tvec_refined = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            return (rvec_refined, tvec_refined)
        else:
            return (rvec, tvec)

    def _compute_reprojection_error(self, object_points: np.ndarray, image_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> float:
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        error = cv2.norm(image_points, projected_points.reshape(-1, 2), cv2.NORM_L2)
        return error / len(image_points)

    def _compute_confidence(self, reprojection_error: float, num_inliers: int, num_points: int) -> float:
        error_confidence = np.exp(-reprojection_error / 2.0)
        inlier_ratio = num_inliers / num_points
        inlier_confidence = inlier_ratio ** 2
        confidence = 0.6 * error_confidence + 0.4 * inlier_confidence
        return np.clip(confidence, 0.0, 1.0)

    def _update_pose_history(self, pose_result: PoseResult):
        if pose_result.success:
            self.pose_history.append(pose_result)
            if len(self.pose_history) > self.max_history_size:
                self.pose_history.pop(0)

    def estimate_pose_uncertainty(self, pose_result: PoseResult) -> Dict[str, np.ndarray]:
        if not pose_result.success:
            return {'position_uncertainty': np.full(3, np.inf), 'orientation_uncertainty': np.full(3, np.inf)}
        position_uncertainty = np.full(3, pose_result.reprojection_error)
        position_uncertainty *= pose_result.distance / 100.0
        orientation_uncertainty = np.full(3, pose_result.reprojection_error * 0.1)
        return {'position_uncertainty': position_uncertainty, 'orientation_uncertainty': orientation_uncertainty}

    def filter_pose_temporal(self, current_pose: PoseResult, alpha: float=0.7) -> PoseResult:
        if len(self.pose_history) < 2:
            return current_pose
        previous_pose = self.pose_history[-2]
        if not previous_pose.success:
            return current_pose
        filtered_translation = alpha * current_pose.translation_vector + (1 - alpha) * previous_pose.translation_vector
        filtered_rotation = self._slerp_rotations(previous_pose.rotation_matrix, current_pose.rotation_matrix, alpha)
        filtered_rvec = matrix_to_rodrigues(filtered_rotation)
        filtered_result = PoseResult(success=current_pose.success, rotation_matrix=filtered_rotation, translation_vector=filtered_translation, rotation_vector=filtered_rvec, reprojection_error=current_pose.reprojection_error, confidence=current_pose.confidence, algorithm=current_pose.algorithm + '_filtered', num_inliers=current_pose.num_inliers, processing_time=current_pose.processing_time)
        return filtered_result

    def _slerp_rotations(self, R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
        r1 = R.from_matrix(R1)
        r2 = R.from_matrix(R2)
        r_interp = r1.slerp(r2, t)
        return r_interp.as_matrix()

    def multi_hypothesis_pose_estimation(self, object_points: np.ndarray, image_points: np.ndarray, algorithms: List[str]=None) -> List[PoseResult]:
        if algorithms is None:
            algorithms = ['SOLVEPNP_EPNP', 'SOLVEPNP_P3P', 'SOLVEPNP_ITERATIVE']
        results = []
        for algorithm in algorithms:
            try:
                result = self.estimate_pose(object_points, image_points, algorithm=algorithm, use_ransac=False, refine=True)
                results.append(result)
            except Exception as e:
                print(f'Algorithm {algorithm} failed: {e}')
        try:
            ransac_result = self.estimate_pose(object_points, image_points, algorithm=self.default_algorithm, use_ransac=True, refine=True)
            ransac_result.algorithm += '_RANSAC'
            results.append(ransac_result)
        except Exception as e:
            print(f'RANSAC failed: {e}')
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def validate_pose(self, pose_result: PoseResult, max_error: float=5.0, min_confidence: float=0.3) -> bool:
        if not pose_result.success:
            return False
        if pose_result.reprojection_error > max_error:
            return False
        if pose_result.confidence < min_confidence:
            return False
        if np.any(np.isnan(pose_result.translation_vector)):
            return False
        if np.any(np.isnan(pose_result.rotation_vector)):
            return False
        distance = pose_result.distance
        if distance < 10.0 or distance > 10000.0:
            return False
        return True

    def get_pose_statistics(self) -> Dict[str, Any]:
        if not self.pose_history:
            return {'num_estimates': 0, 'success_rate': 0.0, 'mean_error': 0.0, 'mean_confidence': 0.0}
        successful_poses = [p for p in self.pose_history if p.success]
        stats = {'num_estimates': len(self.pose_history), 'successful_estimates': len(successful_poses), 'success_rate': len(successful_poses) / len(self.pose_history)}
        if successful_poses:
            errors = [p.reprojection_error for p in successful_poses]
            confidences = [p.confidence for p in successful_poses]
            distances = [p.distance for p in successful_poses]
            stats.update({'mean_error': np.mean(errors), 'std_error': np.std(errors), 'mean_confidence': np.mean(confidences), 'std_confidence': np.std(confidences), 'mean_distance': np.mean(distances), 'std_distance': np.std(distances), 'algorithms_used': list(set((p.algorithm for p in successful_poses)))})
        else:
            stats.update({'mean_error': 0.0, 'std_error': 0.0, 'mean_confidence': 0.0, 'std_confidence': 0.0, 'mean_distance': 0.0, 'std_distance': 0.0, 'algorithms_used': []})
        return stats

    def reset_history(self):
        self.pose_history.clear()
        print('Pose estimation history cleared')