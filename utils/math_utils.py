import numpy as np
import cv2

def rodrigues_to_matrix(rvec):
    return cv2.Rodrigues(rvec)[0]

def matrix_to_rodrigues(R):
    return cv2.Rodrigues(R)[0]

def homogeneous_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def decompose_homogeneous(T):
    R = T[:3, :3]
    t = T[:3, 3:4]
    return (R, t)

def compute_reprojection_error(objpoints, imgpoints, rvec, tvec, K, dist_coeffs):
    projected_points, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist_coeffs)
    error = cv2.norm(imgpoints, projected_points, cv2.NORM_L2) / len(projected_points)
    return error

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    translated = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(translated ** 2, axis=1)))
    scale = np.sqrt(2) / mean_dist
    T = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]])
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (T @ homogeneous.T).T[:, :2]
    return (normalized, T)