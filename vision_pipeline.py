import cv2
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from calibration.camera_calibrator import CameraCalibrator
from aruco_tracking.aruco_detector import ArUcoDetector
from pose_estimation.pose_estimator import PoseEstimator, PoseResult
from visualization.pose_visualizer import PoseVisualizer

@dataclass
class PipelineConfig:
    camera_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    calibration_file: Optional[str] = None
    auto_calibrate: bool = False
    aruco_dictionary: str = 'DICT_4X4_50'
    marker_size: float = 50.0
    pose_algorithm: str = 'SOLVEPNP_EPNP'
    use_ransac: bool = True
    temporal_filtering: bool = True
    filter_alpha: float = 0.3
    show_axes: bool = True
    show_cube: bool = True
    show_trajectory: bool = True
    show_info: bool = True
    max_processing_time: float = 0.033
    enable_threading: bool = True
    save_frames: bool = False
    output_directory: str = 'output'
    save_trajectory: bool = True

@dataclass
class PipelineResult:
    timestamp: float
    success: bool
    frame: np.ndarray
    detection_results: Dict[str, Any]
    pose_results: List[PoseResult]
    processing_time: float
    fps: float

    @property
    def num_markers(self) -> int:
        return self.detection_results.get('num_markers', 0)

    @property
    def best_pose(self) -> Optional[PoseResult]:
        if self.pose_results:
            return max(self.pose_results, key=lambda x: x.confidence)
        return None

class RealTimeVisionPipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.running = False
        self._initialize_components()
        self.frame_count = 0
        self.fps_history = []
        self.processing_times = []
        if config.enable_threading:
            self.frame_queue = queue.Queue(maxsize=2)
            self.result_queue = queue.Queue(maxsize=2)
            self.processing_thread = None
        Path(config.output_directory).mkdir(exist_ok=True)
        self.statistics = {'total_frames': 0, 'successful_frames': 0, 'total_markers_detected': 0, 'mean_processing_time': 0.0, 'mean_fps': 0.0, 'pose_estimation_success_rate': 0.0}

    def _initialize_components(self):
        print('Initializing vision pipeline components...')
        if self.config.calibration_file and Path(self.config.calibration_file).exists():
            print(f'Loading calibration from {self.config.calibration_file}')
            self.calibrator = CameraCalibrator()
            if not self.calibrator.load_calibration(self.config.calibration_file):
                raise RuntimeError('Failed to load calibration')
            camera_matrix = self.calibrator.camera_matrix
            dist_coeffs = self.calibrator.dist_coeffs
        else:
            print('No calibration file found, using default parameters')
            camera_matrix = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros((1, 5), dtype=np.float32)
        self.aruco_detector = ArUcoDetector(dictionary_name=self.config.aruco_dictionary, marker_size=self.config.marker_size)
        self.pose_estimator = PoseEstimator(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, default_algorithm=self.config.pose_algorithm)
        self.visualizer = PoseVisualizer(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print(' All components initialized successfully')

    def initialize_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.config.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open camera {self.config.camera_id}')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS')
        return cap

    def process_frame(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        start_time = time.time()
        try:
            detection_results = self.aruco_detector.detect_markers(frame, self.pose_estimator.camera_matrix, self.pose_estimator.dist_coeffs)
            pose_results = []
            for detection in detection_results['detections']:
                marker_id = detection['id']
                object_points = self.aruco_detector.marker_object_points
                image_points = detection['corners']
                pose_result = self.pose_estimator.estimate_pose(object_points, image_points, algorithm=self.config.pose_algorithm, use_ransac=self.config.use_ransac, refine=True)
                if self.config.temporal_filtering and pose_result.success and (len(self.pose_estimator.pose_history) > 0):
                    pose_result = self.pose_estimator.filter_pose_temporal(pose_result, self.config.filter_alpha)
                if self.pose_estimator.validate_pose(pose_result):
                    pose_results.append(pose_result)
            if pose_results:
                best_pose = max(pose_results, key=lambda x: x.confidence)
                frame = self.visualizer.draw_comprehensive_visualization(frame, best_pose.rotation_vector, best_pose.translation_vector, marker_id=best_pose.algorithm, show_axes=self.config.show_axes, show_cube=self.config.show_cube, show_trajectory=self.config.show_trajectory, show_info=self.config.show_info)
            processing_time = time.time() - start_time
            result = PipelineResult(timestamp=timestamp, success=len(pose_results) > 0, frame=frame, detection_results=detection_results, pose_results=pose_results, processing_time=processing_time, fps=1.0 / processing_time if processing_time > 0 else 0)
            self._update_statistics(result)
            return result
        except Exception as e:
            print(f'Pipeline processing error: {e}')
            processing_time = time.time() - start_time
            return PipelineResult(timestamp=timestamp, success=False, frame=frame, detection_results={'num_markers': 0, 'detections': []}, pose_results=[], processing_time=processing_time, fps=1.0 / processing_time if processing_time > 0 else 0)

    def _update_statistics(self, result: PipelineResult):
        self.statistics['total_frames'] += 1
        if result.success:
            self.statistics['successful_frames'] += 1
        self.statistics['total_markers_detected'] += result.num_markers
        self.processing_times.append(result.processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        self.statistics['mean_processing_time'] = np.mean(self.processing_times)
        self.fps_history.append(result.fps)
        if len(self.fps_history) > 100:
            self.fps_history.pop(0)
        self.statistics['mean_fps'] = np.mean(self.fps_history)
        total_detections = sum((len(dr['detections']) for dr in [result.detection_results]))
        successful_poses = len(result.pose_results)
        if total_detections > 0:
            self.statistics['pose_estimation_success_rate'] = successful_poses / total_detections

    def run_realtime(self, duration: Optional[float]=None) -> None:
        print(f"Starting real-time pipeline (duration: {duration or 'unlimited'}s)")
        print('Controls: SPACE=Save frame, S=Save stats, C=Clear trajectory, Q=Quit')
        cap = self.initialize_camera()
        self.running = True
        start_time = time.time()
        last_stats_time = start_time
        frame_count = 0
        pose_history = []
        try:
            while self.running:
                loop_start = time.time()
                if duration and loop_start - start_time > duration:
                    break
                ret, frame = cap.read()
                if not ret:
                    print('Failed to read frame from camera')
                    break
                timestamp = loop_start
                frame_count += 1
                result = self.process_frame(frame, timestamp)
                if result.pose_results:
                    best_pose = result.best_pose
                    pose_history.append({'timestamp': timestamp, 'rvec': best_pose.rotation_vector.copy(), 'tvec': best_pose.translation_vector.copy(), 'marker_id': frame_count})
                    if len(pose_history) > 500:
                        pose_history.pop(0)
                result.frame = self._add_performance_overlay(result.frame, result)
                cv2.imshow('6-DoF Pose Estimation Pipeline', result.frame)
                key = cv2.waitKey(1) & 255
                if key == ord(' '):
                    self._save_frame(result.frame, frame_count)
                elif key == ord('s'):
                    self._save_statistics()
                    if pose_history:
                        self._save_trajectory(pose_history)
                elif key == ord('c'):
                    self.visualizer.clear_trajectory()
                    self.pose_estimator.reset_history()
                    pose_history.clear()
                    print('Trajectory cleared')
                elif key == ord('q'):
                    break
                processing_time = time.time() - loop_start
                target_time = 1.0 / self.config.camera_fps
                if processing_time < target_time:
                    time.sleep(target_time - processing_time)
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            print('\nSaving final data...')
            self._save_statistics()
            if pose_history:
                self._save_trajectory(pose_history)
            self._print_final_statistics()

    def _add_performance_overlay(self, frame: np.ndarray, result: PipelineResult) -> np.ndarray:
        overlay = frame.copy()
        overlay_height = 120
        overlay_bg = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay_bg[:] = (0, 0, 0)
        y_offset = 25
        lines = [f'FPS: {result.fps:.1f} (Target: {self.config.camera_fps})', f'Processing: {result.processing_time * 1000:.1f}ms', f'Markers: {result.num_markers}', f"Success Rate: {self.statistics['pose_estimation_success_rate'] * 100:.1f}%"]
        for line in lines:
            cv2.putText(overlay_bg, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
        alpha = 0.7
        frame[:overlay_height, :] = cv2.addWeighted(frame[:overlay_height, :], 1 - alpha, overlay_bg, alpha, 0)
        return frame

    def _save_frame(self, frame: np.ndarray, frame_count: int) -> None:
        if self.config.save_frames:
            timestamp = int(time.time() * 1000)
            filename = f'frame_{frame_count:06d}_{timestamp}.jpg'
            filepath = Path(self.config.output_directory) / filename
            cv2.imwrite(str(filepath), frame)
            print(f'Frame saved: {filepath}')

    def _save_statistics(self) -> None:
        stats_file = Path(self.config.output_directory) / 'pipeline_statistics.json'
        stats_copy = self.statistics.copy()
        stats_copy['timestamp'] = time.time()
        stats_copy['config'] = {'camera_width': self.config.camera_width, 'camera_height': self.config.camera_height, 'camera_fps': self.config.camera_fps, 'aruco_dictionary': self.config.aruco_dictionary, 'marker_size': self.config.marker_size, 'pose_algorithm': self.config.pose_algorithm}
        with open(stats_file, 'w') as f:
            json.dump(stats_copy, f, indent=2, default=str)
        print(f'Statistics saved: {stats_file}')

    def _save_trajectory(self, pose_history: List[Dict]) -> None:
        if self.config.save_trajectory and len(pose_history) > 10:
            trajectory_file = Path(self.config.output_directory) / f'trajectory_{int(time.time())}.png'
            self.visualizer.save_trajectory_plot(pose_history, str(trajectory_file))
            data_file = Path(self.config.output_directory) / f'trajectory_data_{int(time.time())}.json'
            trajectory_data = []
            for pose in pose_history:
                trajectory_data.append({'timestamp': pose['timestamp'], 'rvec': pose['rvec'].flatten().tolist(), 'tvec': pose['tvec'].flatten().tolist(), 'marker_id': pose['marker_id']})
            with open(data_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            print(f'Trajectory saved: {trajectory_file}')

    def _print_final_statistics(self) -> None:
        print('\n' + '=' * 60)
        print('PIPELINE PERFORMANCE SUMMARY')
        print('=' * 60)
        stats = self.statistics
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Successful frames: {stats['successful_frames']} ({stats['successful_frames'] / stats['total_frames'] * 100:.1f}%)")
        print(f"Total markers detected: {stats['total_markers_detected']}")
        print(f"Mean processing time: {stats['mean_processing_time'] * 1000:.2f} ms")
        print(f"Mean FPS: {stats['mean_fps']:.1f}")
        print(f"Pose estimation success rate: {stats['pose_estimation_success_rate'] * 100:.1f}%")
        if stats['total_frames'] > 0:
            avg_markers_per_frame = stats['total_markers_detected'] / stats['total_frames']
            print(f'Average markers per frame: {avg_markers_per_frame:.2f}')
        print('=' * 60)

    def run_benchmark(self, duration: float=30.0) -> Dict[str, Any]:
        print(f'Running pipeline benchmark for {duration} seconds...')
        cap = self.initialize_camera()
        benchmark_results = {'duration': duration, 'frames_processed': 0, 'processing_times': [], 'fps_values': [], 'markers_detected': 0, 'pose_success_count': 0, 'pose_attempt_count': 0}
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.process_frame(frame, time.time())
                benchmark_results['frames_processed'] += 1
                benchmark_results['processing_times'].append(result.processing_time)
                benchmark_results['fps_values'].append(result.fps)
                benchmark_results['markers_detected'] += result.num_markers
                if result.detection_results.get('detections'):
                    benchmark_results['pose_attempt_count'] += len(result.detection_results['detections'])
                    benchmark_results['pose_success_count'] += len(result.pose_results)
        finally:
            cap.release()
        if benchmark_results['frames_processed'] > 0:
            processing_times = np.array(benchmark_results['processing_times'])
            fps_values = np.array(benchmark_results['fps_values'])
            benchmark_results.update({'mean_processing_time': np.mean(processing_times), 'std_processing_time': np.std(processing_times), 'min_processing_time': np.min(processing_times), 'max_processing_time': np.max(processing_times), 'mean_fps': np.mean(fps_values), 'std_fps': np.std(fps_values), 'min_fps': np.min(fps_values), 'max_fps': np.max(fps_values), 'actual_duration': time.time() - start_time, 'achieved_fps': benchmark_results['frames_processed'] / (time.time() - start_time)})
        if benchmark_results['pose_attempt_count'] > 0:
            benchmark_results['pose_success_rate'] = benchmark_results['pose_success_count'] / benchmark_results['pose_attempt_count']
        print('\nBenchmark Results:')
        print(f"  Duration: {benchmark_results['actual_duration']:.2f}s")
        print(f"  Frames processed: {benchmark_results['frames_processed']}")
        print(f"  Achieved FPS: {benchmark_results['achieved_fps']:.1f}")
        print(f"  Mean processing time: {benchmark_results['mean_processing_time'] * 1000:.2f}ms")
        print(f"  Pose success rate: {benchmark_results.get('pose_success_rate', 0) * 100:.1f}%")
        return benchmark_results

    def stop(self):
        self.running = False
        print('Pipeline stopped')