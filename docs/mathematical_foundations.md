# Mathematical Foundations of 6-DoF Pose Estimation

## 1. Camera Intrinsic Matrix

### Mathematical Intuition
The camera intrinsic matrix (K) describes the internal geometry of the camera and how 3D points are projected onto the 2D image plane.

```
K = [fx  s  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```

**Parameters:**
- **fx, fy**: Focal lengths in pixel units (horizontal and vertical)
- **cx, cy**: Principal point (optical center) in pixel coordinates
- **s**: Skew coefficient (typically 0 for modern cameras)

### Practical Implementation
```python
K = np.array([[fx, 0,   cx],
              [0,  fy,  cy],
              [0,  0,   1]], dtype=np.float64)
```

**Why it's needed:** Without accurate intrinsics, we cannot correctly map 3D world coordinates to 2D image coordinates, making pose estimation impossible.

---

## 2. Lens Distortion Models

### Radial Distortion
Lens curvature causes straight lines to appear curved. Radial distortion models this effect:

```
x_corrected = x * (1 + k1*r² + k2*r⁴ + k3*r⁶)
y_corrected = y * (1 + k1*r² + k2*r⁴ + k3*r⁶)
```

Where r² = x² + y², and k1, k2, k3 are radial distortion coefficients.

### Tangential Distortion
Caused by imperfect lens alignment:

```
x_corrected = x + 2*p1*x*y + p2*(r² + 2*x²)
y_corrected = y + p1*(r² + 2*y²) + 2*p2*x*y
```

Where p1, p2 are tangential distortion coefficients.

### Practical Implementation
```python
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
```

**Why it's needed:** Uncorrected distortion leads to systematic errors in pose estimation, especially at image edges.

---

## 3. Checkerboard Calibration Principle

### Mathematical Foundation
Checkerboard calibration exploits the fact that:
1. Checkerboard corners lie on a planar surface
2. The pattern has known geometry (equal square sizes)
3. Multiple views provide constraints to solve for camera parameters

### Homography Relationship
For a planar calibration pattern:
```
s * [u v 1]ᵀ = K * [r1 r2 t] * [X Y 1]ᵀ
```
Where:
- [u v]ᵀ are image coordinates
- [X Y]ᵀ are object coordinates on the calibration plane
- [r1 r2 t] are the first two columns of the rotation matrix and translation

**Why it's needed:** This relationship allows us to estimate camera parameters from known 3D-2D correspondences.

---

## 4. Reprojection Error

### Mathematical Definition
Reprojection error measures the difference between projected 3D points and detected 2D points:

```
error = ||u_detected - u_projected||²
```

Where u_projected = K * [R|t] * X_world (projected through the pinhole camera model).

### Practical Implementation
```python
reprojection_error = cv2.norm(imgpoints, projected_points, cv2.NORM_L2) / len(projected_points)
```

**Why it's needed:** It's our primary metric for calibration quality. < 0.5 pixels indicates excellent calibration.

---

## 5. Rotation Representations

### Rotation Matrix (3x3)
```
R = [r11 r12 r13]
    [r21 r22 r23]
    [r31 r32 r33]
```

Properties: RᵀR = I, det(R) = 1

### Rodrigues Representation
Converts between rotation matrix and rotation vector:

```
θ = ||r||  (rotation angle)
k = r/||r||  (rotation axis)

R = I*cos(θ) + (1-cos(θ))*k*kᵀ + sin(θ)*[k]×
```

Where [k]× is the skew-symmetric matrix of k.

**Why it's needed:** Rotation vectors are more compact (3 parameters vs 9) and avoid gimbal lock issues.

---

## 6. Perspective Projection

### Pinhole Camera Model
```
s * [u v 1]ᵀ = K * [R|t] * [X Y Z 1]ᵀ
```

Breaking it down:
1. Transform to camera coordinates: X_cam = R * X_world + t
2. Project to image plane: x = X/Z, y = Y/Z
3. Apply intrinsics: u = fx*x + cx, v = fy*y + cy

### Practical Implementation
```python
# Project 3D points to 2D
projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
```

**Why it's needed:** This is the fundamental relationship between 3D world and 2D image coordinates.

---

## 7. Perspective-n-Point (PnP) Algorithms

### Problem Statement
Given n 3D-2D correspondences, find the camera pose (R, t) that minimizes reprojection error.

### P3P Algorithm
- Uses exactly 3 point correspondences
- Provides up to 4 solutions
- Minimal solver, computationally efficient

### EPnP Algorithm
- Uses n ≥ 4 points
- Represents the solution as weighted combination of 4 control points
- More stable than P3P

### RANSAC PnP
- Robust to outliers
- Randomly samples minimal subsets
- Iteratively finds consensus

**Why it's needed:** PnP is the core algorithm that estimates camera pose from known correspondences.

---

## 8. Coordinate Frame Transformations

### Homogeneous Transformation Matrix
```
T = [R t]
    [0 1]
```

Where T is 4x4, combining rotation and translation.

### Transform Composition
```
T_AB = T_AC * T_CB
```

**Why it's needed:** Allows us to transform between different coordinate frames (camera, world, marker).

---

## 9. Implementation Considerations

### Numerical Stability
- Use double precision for critical calculations
- Normalize coordinates when possible
- Handle degenerate cases (collinear points)

### Optimization
- Subpixel corner refinement
- Iterative refinement (Levenberg-Marquardt)
- Multi-threading for real-time performance

### Robustness
- RANSAC for outlier rejection
- Temporal filtering (Kalman filter)
- Multi-marker tracking

These mathematical foundations form the theoretical backbone of our 6-DoF pose estimation system. Understanding these concepts is crucial for debugging and optimizing the implementation.
