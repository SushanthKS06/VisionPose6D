import cv2
import numpy as np

def resize_with_aspect_ratio(image, width=None, height=None):
    if width is None and height is None:
        return image
    h, w = image.shape[:2]
    if width is not None:
        ratio = width / w
        new_h = int(h * ratio)
        new_w = width
    else:
        ratio = height / h
        new_h = height
        new_w = int(w * ratio)
    return cv2.resize(image, (new_w, new_h))

def draw_checkerboard_corners(image, corners, pattern_size, found):
    result = image.copy()
    if found:
        cv2.drawChessboardCorners(result, pattern_size, corners, found)
        for i, corner in enumerate(corners.reshape(-1, 2)):
            cv2.putText(result, str(i), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    else:
        cv2.putText(result, 'Pattern not found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return result

def undistort_image(image, K, dist_coeffs):
    return cv2.undistort(image, K, dist_coeffs)

def create_side_by_side_comparison(distorted, undistorted, title1='Distorted', title2='Undistorted'):
    h1, w1 = distorted.shape[:2]
    h2, w2 = undistorted.shape[:2]
    if h1 != h2:
        if h1 > h2:
            distorted = resize_with_aspect_ratio(distorted, height=h2)
        else:
            undistorted = resize_with_aspect_ratio(undistorted, height=h1)
    distorted_with_title = distorted.copy()
    undistorted_with_title = undistorted.copy()
    cv2.putText(distorted_with_title, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(undistorted_with_title, title2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.hstack([distorted_with_title, undistorted_with_title])

def enhance_contrast(image, method='clahe'):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    else:
        enhanced = cv2.equalizeHist(gray)
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced