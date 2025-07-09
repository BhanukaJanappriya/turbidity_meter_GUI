# analysis_utils.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred, original

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

def apply_media_filter_if_needed(gray):
    if detect_salt_pepper_noise(gray):
        return cv2.medianBlur(gray, (3, 3)), True
    return gray, False


def compute_intensity_metrics(image):
    return np.mean(image), np.var(image), np.min(image), np.max(image)

def compute_turbidity_index(min_intensity, max_intensity):
    return (max_intensity - min_intensity) / (max_intensity + min_intensity + 1e-5)

def compute_edge_density(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges > 0) / (image.shape[0] * image.shape[1])

def calculate_turbidity_from_red_channel(image):
    h, w, _ = image.shape
    center_x, center_y = h // 2, w // 2
    crop_size = 50
    cropped = image[center_x - crop_size:center_x + crop_size,
                    center_y - crop_size:center_y + crop_size]
    m_red = np.mean(cropped[:, :, 2])
    turb = -123.03 * np.exp(-m_red / 202.008) - 184.47115 * np.exp(-m_red / 1157.359) + 313.5892
    turbidity_out = round(-10.03 * turb + 1274.35)
    return turbidity_out, m_red, turb

def save_red_channel_histogram(image, output_path):
    red_channel = image[:, :, 2]  # OpenCV is BGR
    plt.figure(figsize=(5, 3))
    plt.hist(red_channel.flatten(), bins=50, color='red', alpha=0.8)
    plt.title("Red Channel Histogram")
    plt.xlabel("Red Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    

