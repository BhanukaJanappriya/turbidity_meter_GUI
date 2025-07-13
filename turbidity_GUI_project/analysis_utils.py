# analysis_utils.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from noise_utils import remove_watermark_if_needed, apply_median_filter_if_needed


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Watermark Removal (if present)
    cleaned_image, watermark_removed = remove_watermark_if_needed(image)
    
    # Grayscale conversion
    gray = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
    
    # Salt-and-pepper noise removal (if present)
    denoised_gray, noise_removed = apply_median_filter_if_needed(gray)
    
    # Gaussian blur for smoothing
    blurred = cv2.GaussianBlur(denoised_gray, (5, 5), 0)

    # print or log status
    if watermark_removed:
        print("[INFO] Watermark removed")
    if noise_removed:
        print("[INFO] Salt-and-pepper noise removed")

    return blurred, cleaned_image

def compute_intensity_metrics(image):
    return np.mean(image), np.var(image), np.min(image), np.max(image)

def compute_turbidity_index(min_intensity, max_intensity):
    return (float(max_intensity) - float(min_intensity)) / (float(max_intensity) + float(min_intensity) + 1e-5)

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
    turb = -0.0063 * (m_red**2) - 3.3426 * m_red + 1104.4
    turbidity_out = round(turb)
    if turbidity_out < 0:
        turbidity_out = 0
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
    
