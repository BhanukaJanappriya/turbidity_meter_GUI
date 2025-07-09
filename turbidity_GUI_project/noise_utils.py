import cv2
import numpy as np

def detect_watermark(gray):
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return coverage > 0.01, mask  # Apply only if more than 1% of image has watermark

def remove_watermark_if_needed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    has_watermark, mask = detect_watermark(gray)
    if has_watermark:
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA), True
    return image, False

def detect_salt_pepper_noise(gray):
    # Count extreme pixels
    total_pixels = gray.size
    black_pixels = np.sum(gray == 0)
    white_pixels = np.sum(gray == 255)
    noisy_ratio = (black_pixels + white_pixels) / total_pixels
    return noisy_ratio > 0.01  # Only filter if more than 1% affected

def apply_median_filter_if_needed(gray):
    if detect_salt_pepper_noise(gray):
        return cv2.medianBlur(gray, 3), True
    return gray, False
