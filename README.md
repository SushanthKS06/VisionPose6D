# 6-DoF Pose Estimation and Camera Calibration Pipeline

A production-grade computer vision system for real-time 6-DoF pose estimation and camera calibration, built with Python, OpenCV, and NumPy. This system is designed for robotics, augmented reality, and computer vision applications requiring high-precision pose tracking.

## 🎯 Project Overview

This complete pipeline provides:
- **Camera Calibration**: High-accuracy calibration using checkerboard patterns
- **Distortion Correction**: Lens distortion modeling and correction
- **ArUco Marker Tracking**: Robust marker detection and identification
- **6-DoF Pose Estimation**: Multiple PnP algorithms with uncertainty quantification
- **Real-Time Visualization**: 3D axes, object projection, and trajectory tracking
- **Robustness Features**: Kalman filtering, outlier detection, and adaptive noise handling

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd VisionPose6D

# Install dependencies
pip install -r requirements.txt
```

### 2. Camera Calibration

```bash
# Live camera calibration
python main.py --calibrate --mode live --camera 0

# Or calibrate from existing images
python main.py --calibrate --mode images --path calibration_images/
```

### 3. Generate ArUco Markers

```bash
# Generate markers 0-4
python main.py --generate-markers --ids 0 1 2 3 4 --output markers/
```

### 4. Run Real-Time Pose Estimation

```bash
# Run with calibrated camera
python main.py --run --calibration camera_calibration.pkl

# Or run benchmark
python main.py --benchmark --calibration camera_calibration.pkl --duration 30
```

## 📁 Project Structure

```
VisionPose6D/
├── calibration/                 # Camera calibration module
│   ├── __init__.py
│   └── camera_calibrator.py
├── aruco_tracking/             # ArUco marker detection
│   ├── __init__.py
│   └── aruco_detector.py
├── pose_estimation/            # 6-DoF pose estimation
│   ├── __init__.py
│   └── pose_estimator.py
├── visualization/              # 3D visualization
│   ├── __init__.py
│   └── pose_visualizer.py
├── robustness/                 # Robustness features
│   ├── __init__.py
│   └── kalman_filter.py
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── math_utils.py
│   └── image_utils.py
├── scripts/                    # Demonstration scripts
│   ├── calibrate_camera.py
│   ├── aruco_demo.py
│   ├── pose_estimation_demo.py
│   ├── visualization_demo.py
│   ├── distortion_correction_demo.py
│   └── robustness_demo.py
├── docs/                       # Documentation
│   └── mathematical_foundations.md
├── main.py                     # Main entry point
├── vision_pipeline.py          # Real-time pipeline
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 System Architecture

### Pipeline Flow

```
Camera Capture → Distortion Correction → ArUco Detection → Pose Estimation → Visualization
     ↓                    ↓                    ↓              ↓              ↓
  Calibration         Lens Model        Marker ID       PnP Solver    3D Rendering
```

### Core Components

1. **Camera Calibration Module** (`calibration/`)
   - Checkerboard corner detection with subpixel accuracy
   - Camera intrinsic matrix estimation
   - Lens distortion coefficient calculation
   - Reproduction error analysis and validation

2. **ArUco Tracking Module** (`aruco_tracking/`)
   - Multi-dictionary marker detection
   - Subpixel corner refinement
   - Marker tracking across frames
   - Quality assessment and validation

3. **Pose Estimation Module** (`pose_estimation/`)
   - Multiple PnP algorithms (EPnP, P3P, DLS, UPnP)
   - RANSAC-based robust estimation
   - Uncertainty quantification
   - Multi-hypothesis tracking

4. **Visualization Module** (`visualization/`)
   - 3D coordinate axes rendering
   - 3D object projection (cubes, models)
   - Trajectory tracking and plotting
   - Real-time performance optimization

5. **Robustness Module** (`robustness/`)
   - Kalman filtering for pose smoothing
   - Outlier detection and rejection
   - Adaptive noise estimation
   - Occlusion handling and prediction

## 📊 Mathematical Foundations

### Camera Model

The system uses the pinhole camera model with lens distortion:

```
s * [u v 1]ᵀ = K * [R|t] * [X Y Z 1]ᵀ
```

Where:
- `K` is the camera intrinsic matrix
- `[R|t]` represents the 6-DoF pose
- `[X Y Z]` are 3D world coordinates
- `[u v]` are 2D image coordinates

### Distortion Model

Radial and tangential distortion correction:

```
x_corrected = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
y_corrected = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
```

### PnP Algorithms

The system implements multiple Perspective-n-Point algorithms:

- **EPnP**: Efficient PnP for n ≥ 4 points
- **P3P**: Minimal solver for exactly 3 points
- **RANSAC**: Robust estimation with outlier rejection
- **DLS**: Direct Least Squares method
- **UPnP**: Uncalibrated PnP (when focal length unknown)

## 🎮 Usage Examples

### Basic Calibration

```python
from calibration.camera_calibrator import CameraCalibrator

# Initialize calibrator
calibrator = CameraCalibrator(pattern_size=(9, 6), square_size=25.0)

# Add calibration images
for image_path in calibration_images:
    image = cv2.imread(image_path)
    calibrator.add_calibration_image(image)

# Perform calibration
results = calibrator.calibrate()
print(f"Reprojection error: {results['reprojection_error']['mean']:.3f} pixels")

# Save calibration
calibrator.save_calibration("camera_calibration.pkl")
```

### Real-Time Pose Estimation

```python
from vision_pipeline import RealTimeVisionPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    camera_id=0,
    camera_width=1280,
    camera_height=720,
    aruco_dictionary="DICT_4X4_50",
    marker_size=50.0,
    calibration_file="camera_calibration.pkl"
)

# Create and run pipeline
pipeline = RealTimeVisionPipeline(config)
pipeline.run_realtime()
```

### Advanced Pose Estimation

```python
from pose_estimation.pose_estimator import PoseEstimator

# Initialize estimator
estimator = PoseEstimator(camera_matrix, dist_coeffs)

# Multi-hypothesis pose estimation
results = estimator.multi_hypothesis_pose_estimation(
    object_points, image_points,
    algorithms=["SOLVEPNP_EPNP", "SOLVEPNP_P3P", "SOLVEPNP_ITERATIVE"]
)

# Get best result
best_pose = results[0]
print(f"Position: {best_pose.position}")
print(f"Rotation (deg): {best_pose.euler_angles}")
print(f"Confidence: {best_pose.confidence}")
```

### Robust Filtering

```python
from robustness.kalman_filter import AdaptivePoseFilter

# Initialize adaptive filter
adaptive_filter = AdaptivePoseFilter()

# Process measurements with outlier detection
result = adaptive_filter.process_measurement(
    position, rotation, confidence=0.8
)

if result['is_outlier']:
    print("Outlier detected and rejected")
elif result['is_predicted']:
    print("Pose predicted during occlusion")
else:
    print(f"Filtered pose: {result['filtered_position']}")
```

## 📈 Performance Optimization

### Real-Time Optimization

1. **Precomputed Distortion Maps**: For fast undistortion
2. **Subpixel Corner Refinement**: For improved accuracy
3. **Multi-threading**: For parallel processing
4. **Memory Pooling**: To reduce allocations
5. **Early Rejection**: For invalid measurements

### Accuracy Improvements

1. **Temporal Filtering**: Kalman smoothing
2. **Multi-algorithm Consensus**: Best-of-N approaches
3. **Adaptive Thresholds**: Dynamic parameter tuning
4. **Outlier Rejection**: Statistical filtering
5. **Uncertainty Quantification**: Error bounds

## 🔍 Calibration Guidelines

### Camera Calibration Best Practices

1. **Image Collection**:
   - Use 15-20 different views
   - Cover entire field of view
   - Vary distance and angle
   - Ensure good lighting

2. **Checkerboard Requirements**:
   - Flat, rigid board
   - High contrast pattern
   - Known square size
   - At least 6x9 corners

3. **Calibration Quality**:
   - Target reprojection error < 0.5 pixels
   - Check for systematic errors
   - Validate with test images

### ArUco Marker Guidelines

1. **Marker Selection**:
   - Choose appropriate dictionary size
   - Consider detection distance
   - Account for printing resolution

2. **Physical Setup**:
   - Flat, rigid markers
   - Known marker dimensions
   - Good lighting conditions
   - Minimal reflections

## 🛠️ Troubleshooting

### Common Issues

**High Reprojection Error (> 1.0 pixels)**:
- Check checkerboard flatness
- Improve lighting conditions
- Increase number of calibration images
- Verify square size measurements

**Marker Detection Failures**:
- Check dictionary compatibility
- Adjust detection parameters
- Improve image contrast
- Verify marker print quality

**Pose Estimation Instability**:
- Enable temporal filtering
- Use RANSAC for robustness
- Check calibration quality
- Reduce motion blur

**Performance Issues**:
- Reduce image resolution
- Disable unnecessary visualizations
- Use precomputed distortion maps
- Optimize detection parameters

### Debug Mode

Enable debug information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use built-in statistics
stats = estimator.get_pose_statistics()
print(f"Success rate: {stats['success_rate']*100:.1f}%")
```

## 📚 API Reference

### Core Classes

- `CameraCalibrator`: Camera calibration with checkerboards
- `ArUcoDetector`: ArUco marker detection and tracking
- `PoseEstimator`: 6-DoF pose estimation with multiple algorithms
- `PoseVisualizer`: 3D visualization and rendering
- `RealTimeVisionPipeline`: Complete real-time system
- `AdaptivePoseFilter`: Robust pose filtering

### Key Methods

#### CameraCalibrator
```python
add_calibration_image(image)  # Add calibration image
calibrate()                  # Perform calibration
save_calibration(path)       # Save calibration data
get_calibration_quality()    # Quality assessment
```

#### ArUcoDetector
```python
detect_markers(image, K, dist)  # Detect markers
estimate_pose(corners, K, dist)  # Estimate pose
draw_detections(image, results) # Visualize results
```

#### PoseEstimator
```python
estimate_pose(obj_pts, img_pts)  # Single pose estimation
multi_hypothesis_pose_estimation() # Multiple algorithms
validate_pose(pose_result)        # Quality validation
```

## 🧪 Testing and Validation

### Unit Tests

```bash
# Run calibration tests
python -m pytest tests/test_calibration.py

# Run pose estimation tests
python -m pytest tests/test_pose_estimation.py

# Run robustness tests
python -m pytest tests/test_robustness.py
```

### Performance Benchmarks

```bash
# Run comprehensive benchmark
python main.py --benchmark --duration 60

# Algorithm comparison
python scripts/pose_estimation_demo.py --synthetic
```

### Validation Datasets

The system has been validated with:
- Synthetic pose sequences
- Standard calibration datasets
- Real-world robotics scenarios
- AR/VR use cases

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd VisionPose6D

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include unit tests for new features

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request with description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision libraries
- ArUco library for marker detection
- NumPy for numerical computations
- Matplotlib for visualization

## 📞 Support

For questions, bug reports, or feature requests:
1. Check the troubleshooting section
2. Search existing issues
3. Create new issue with detailed description
4. Include system information and error logs

## 🔮 Future Development

Planned enhancements:
- Deep learning-based marker detection
- Multi-camera calibration
- SLAM integration
- GPU acceleration
- ROS2 integration
- Web-based interface

---

**Built with ❤️ for the computer vision community**
