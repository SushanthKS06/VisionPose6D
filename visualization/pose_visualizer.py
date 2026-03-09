import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..utils.math_utils import rodrigues_to_matrix, homogeneous_transform

class PoseVisualizer:

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, axis_length: float=50.0, cube_size: float=50.0):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.axis_length = axis_length
        self.cube_size = cube_size
        self.colors = {'x_axis': (0, 0, 255), 'y_axis': (0, 255, 0), 'z_axis': (255, 0, 0), 'origin': (255, 255, 255), 'cube_edges': (255, 255, 0), 'trajectory': (0, 255, 255), 'text': (255, 255, 255), 'background': (0, 0, 0)}
        self.trajectory_points = []
        self.max_trajectory_points = 100
        self._precompute_objects()

    def _precompute_objects(self):
        self.axis_points = np.array([[0, 0, 0], [self.axis_length, 0, 0], [0, self.axis_length, 0], [0, 0, self.axis_length]], dtype=np.float32)
        half_size = self.cube_size / 2.0
        self.cube_vertices = np.array([[-half_size, -half_size, -half_size], [half_size, -half_size, -half_size], [half_size, half_size, -half_size], [-half_size, half_size, -half_size], [-half_size, -half_size, half_size], [half_size, -half_size, half_size], [half_size, half_size, half_size], [-half_size, half_size, half_size]], dtype=np.float32)
        self.cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        self.cube_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]

    def draw_coordinate_axes(self, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, thickness: int=3) -> np.ndarray:
        result = image.copy()
        projected_points, _ = cv2.projectPoints(self.axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        projected_points = projected_points.reshape(-1, 2).astype(int)
        origin = projected_points[0]
        axis_labels = ['X', 'Y', 'Z']
        for i, (point, color, label) in enumerate(zip(projected_points[1:], [self.colors['x_axis'], self.colors['y_axis'], self.colors['z_axis']], axis_labels)):
            cv2.line(result, origin, point, color, thickness)
            label_offset = 10
            cv2.putText(result, label, (point[0] + label_offset, point[1] - label_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(result, origin, 5, self.colors['origin'], -1)
        return result

    def draw_cube(self, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, filled: bool=False, thickness: int=2) -> np.ndarray:
        result = image.copy()
        projected_vertices, _ = cv2.projectPoints(self.cube_vertices, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        projected_vertices = projected_vertices.reshape(-1, 2).astype(int)
        if filled:
            for face in self.cube_faces:
                face_points = projected_vertices[face]
                if self._is_face_visible(face_points):
                    cv2.fillPoly(result, [face_points], self.colors['cube_edges'])
            for edge in self.cube_edges:
                pt1, pt2 = (projected_vertices[edge[0]], projected_vertices[edge[1]])
                cv2.line(result, pt1, pt2, self.colors['x_axis'], thickness)
        else:
            for edge in self.cube_edges:
                pt1, pt2 = (projected_vertices[edge[0]], projected_vertices[edge[1]])
                cv2.line(result, pt1, pt2, self.colors['cube_edges'], thickness)
        return result

    def _is_face_visible(self, face_points: np.ndarray) -> bool:
        v1 = face_points[1] - face_points[0]
        v2 = face_points[2] - face_points[0]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        return cross_product > 0

    def draw_trajectory(self, image: np.ndarray, current_position: np.ndarray, max_points: Optional[int]=None) -> np.ndarray:
        result = image.copy()
        self.trajectory_points.append(current_position.copy())
        if max_points is None:
            max_points = self.max_trajectory_points
        if len(self.trajectory_points) > max_points:
            self.trajectory_points = self.trajectory_points[-max_points:]
        if len(self.trajectory_points) > 1:
            trajectory_array = np.array(self.trajectory_points)
            dummy_rvec = np.zeros((3, 1), dtype=np.float32)
            for i in range(len(trajectory_array) - 1):
                tvec1 = trajectory_array[i].reshape(3, 1)
                tvec2 = trajectory_array[i + 1].reshape(3, 1)
                pt1, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), dummy_rvec, tvec1, self.camera_matrix, self.dist_coeffs)
                pt2, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), dummy_rvec, tvec2, self.camera_matrix, self.dist_coeffs)
                pt1 = pt1[0][0].astype(int)
                pt2 = pt2[0][0].astype(int)
                alpha = (i + 1) / len(trajectory_array)
                color = tuple((int(c * alpha) for c in self.colors['trajectory']))
                cv2.line(result, pt1, pt2, color, 2)
        return result

    def draw_pose_info(self, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, marker_id: Optional[int]=None, position: Tuple[int, int]=(10, 60)) -> np.ndarray:
        result = image.copy()
        rotation_matrix = rodrigues_to_matrix(rvec)
        position_vec = tvec.flatten()
        euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
        distance = np.linalg.norm(position_vec)
        lines = []
        if marker_id is not None:
            lines.append(f'Marker ID: {marker_id}')
        lines.extend([f'Position (mm): [{position_vec[0]:6.1f}, {position_vec[1]:6.1f}, {position_vec[2]:6.1f}]', f'Distance: {distance:6.1f} mm', f'Rotation (deg): [{euler_angles[0]:6.1f}, {euler_angles[1]:6.1f}, {euler_angles[2]:6.1f}]'])
        y_offset = position[1]
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result, (position[0] - 5, y_offset - text_height - 5), (position[0] + text_width + 5, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(result, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            y_offset += text_height + 10
        return result

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

    def draw_comprehensive_visualization(self, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, marker_id: Optional[int]=None, show_axes: bool=True, show_cube: bool=True, show_trajectory: bool=True, show_info: bool=True) -> np.ndarray:
        result = image.copy()
        if show_axes:
            result = self.draw_coordinate_axes(result, rvec, tvec)
        if show_cube:
            result = self.draw_cube(result, rvec, tvec, filled=False)
        if show_trajectory:
            result = self.draw_trajectory(result, tvec)
        if show_info:
            result = self.draw_pose_info(result, rvec, tvec, marker_id)
        return result

    def create_3d_plot(self, poses: List[Dict[str, np.ndarray]], title: str='3D Pose Trajectory', show_axes: bool=True, show_trajectory: bool=True) -> plt.Figure:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        positions = np.array([pose['tvec'].flatten() for pose in poses])
        if show_trajectory and len(positions) > 1:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'c-', linewidth=2, label='Trajectory')
        if len(positions) > 0:
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, label='Current Position')
        if show_axes and len(poses) > 0:
            current_pose = poses[-1]
            rvec = current_pose['rvec']
            tvec = current_pose['tvec']
            rotation_matrix = rodrigues_to_matrix(rvec)
            transformed_axes = rotation_matrix @ self.axis_points.T + tvec.flatten()
            axis_colors = ['red', 'green', 'blue']
            axis_labels = ['X', 'Y', 'Z']
            for i in range(1, 4):
                ax.plot([tvec[0, 0], transformed_axes[0, i]], [tvec[1, 0], transformed_axes[1, i]], [tvec[2, 0], transformed_axes[2, i]], color=axis_colors[i - 1], linewidth=3, label=f'{axis_labels[i - 1]}-axis')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        max_range = np.array([positions[:, 0].max() - positions[:, 0].min(), positions[:, 1].max() - positions[:, 1].min(), positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.legend()
        ax.grid(True)
        return fig

    def clear_trajectory(self):
        self.trajectory_points.clear()

    def save_trajectory_plot(self, poses: List[Dict[str, np.ndarray]], output_path: str, title: str='3D Pose Trajectory') -> bool:
        try:
            fig = self.create_3d_plot(poses, title)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Trajectory plot saved to: {output_path}')
            return True
        except Exception as e:
            print(f'Failed to save trajectory plot: {e}')
            return False

    def create_visualization_grid(self, images: List[np.ndarray], titles: List[str], grid_size: Tuple[int, int]=(2, 2)) -> np.ndarray:
        rows, cols = grid_size
        h, w = images[0].shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i, (image, title) in enumerate(zip(images, titles)):
            if i >= rows * cols:
                break
            row = i // cols
            col = i % cols
            img_resized = cv2.resize(image, (w, h))
            cv2.putText(img_resized, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img_resized
        return grid