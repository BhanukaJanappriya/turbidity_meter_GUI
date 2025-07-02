import tkinter as tk
from tkinter import filedialog, messagebox
from reportlab.pdfgen import canvas
from reportlab.lib.colors import green, yellow, red
from datetime import datetime
import cv2
import os
import matplotlib.pyplot as plt
from reportlab.lib.utils import ImageReader
import tempfile
import numpy as np
from scipy import ndimage
from skimage import measure, filters

from analysis_utils import (
    preprocess_image, compute_intensity_metrics,
    compute_turbidity_index, compute_edge_density,
    calculate_turbidity_from_red_channel, save_red_channel_histogram
)

def save_red_channel_histogram_fixed(image, output_path):
    """Fixed version of histogram generation"""
    red_channel = image[:, :, 2]  # OpenCV is BGR
    plt.figure(figsize=(8, 4))
    plt.hist(red_channel.flatten(), bins=50, color='red', alpha=0.7, edgecolor='darkred')
    plt.title("Red Channel Histogram", fontsize=14, fontweight='bold')
    plt.xlabel("Red Pixel Value (0-255)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Important: close the figure to free memory

def analyze_particle_distribution(image):
    """Analyze particle distribution using connected component analysis"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to identify particles with error handling
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels_im = cv2.connectedComponents(binary)
        
        # Calculate particle statistics
        particle_areas = []
        for label in range(1, min(num_labels, 1000)):  # Limit to prevent memory issues
            area = np.sum(labels_im == label)
            if area > 0:  # Only include valid areas
                particle_areas.append(area)
        
        if particle_areas and len(particle_areas) > 0:
            avg_particle_size = float(np.mean(particle_areas))
            particle_density = len(particle_areas) / (image.shape[0] * image.shape[1])
            std_areas = float(np.std(particle_areas))
            mean_areas = float(np.mean(particle_areas))
            particle_uniformity = 1 - (std_areas / (mean_areas + 1e-5))
        else:
            avg_particle_size = 0.0
            particle_density = 0.0
            particle_uniformity = 1.0
        
        return {
            'particle_count': len(particle_areas),
            'avg_particle_size': round(avg_particle_size, 2),
            'particle_density': round(particle_density * 1000000, 4),  # particles per million pixels
            'particle_uniformity': round(float(particle_uniformity), 4)
        }
    
    except Exception as e:
        print(f"Particle analysis failed: {e}")
        return {
            'particle_count': 0,
            'avg_particle_size': 0.0,
            'particle_density': 0.0,
            'particle_uniformity': 1.0
        }

def compute_texture_features(image):
    """Compute texture features using Local Binary Patterns and Gabor filters"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simplified Local Binary Pattern with fixed range
    def local_binary_pattern_simple(image, radius=1, n_points=8):
        """Simplified LBP that stays within uint8 bounds"""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        # Define neighbor offsets for 8-point circular pattern
        offsets = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dy = int(round(radius * np.sin(angle)))
            dx = int(round(radius * np.cos(angle)))
            offsets.append((dy, dx))
        
        # Calculate LBP
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                binary_num = 0
                for k, (dy, dx) in enumerate(offsets):
                    if image[i + dy, j + dx] >= center:
                        binary_num |= (1 << k)
                lbp[i, j] = binary_num
        
        return lbp
    
    try:
        lbp = local_binary_pattern_simple(gray)
        lbp_variance = np.var(lbp.astype(np.float32))
    except Exception as e:
        print(f"LBP calculation failed: {e}")
        lbp_variance = 0.0
    
    # Gabor filter response with error handling
    try:
        gabor_responses = []
        for theta in [0, 45, 90, 135]:  # Different orientations
            gabor_real, _ = filters.gabor(gray, frequency=0.1, theta=np.deg2rad(theta))
            gabor_responses.append(np.mean(np.abs(gabor_real)))
        
        texture_energy = np.mean(gabor_responses) if gabor_responses else 0.0
        texture_contrast = np.std(gabor_responses) if len(gabor_responses) > 1 else 0.0
    except Exception as e:
        print(f"Gabor filter calculation failed: {e}")
        texture_energy = 0.0
        texture_contrast = 0.0
    
    return {
        'lbp_variance': round(float(lbp_variance), 2),
        'texture_energy': round(float(texture_energy), 4),
        'texture_contrast': round(float(texture_contrast), 4)
    }

def analyze_color_distribution(image):
    """Analyze color distribution across different channels"""
    try:
        # Convert to different color spaces with error handling
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics with safe float conversion
        color_stats = {
            'blue_mean': round(float(np.mean(image[:, :, 0])), 2),
            'green_mean': round(float(np.mean(image[:, :, 1])), 2),
            'red_mean': round(float(np.mean(image[:, :, 2])), 2),
            'hue_std': round(float(np.std(hsv[:, :, 0])), 2),
            'saturation_mean': round(float(np.mean(hsv[:, :, 1])), 2),
            'lightness_std': round(float(np.std(lab[:, :, 0])), 2),
        }
        
        # Calculate color uniformity safely
        image_std = float(np.std(image))
        image_mean = float(np.mean(image))
        color_uniformity = 1 - (image_std / (image_mean + 1e-5))
        color_stats['color_uniformity'] = round(color_uniformity, 4)
        
        return color_stats
    
    except Exception as e:
        print(f"Color analysis failed: {e}")
        return {
            'blue_mean': 0.0,
            'green_mean': 0.0,
            'red_mean': 0.0,
            'hue_std': 0.0,
            'saturation_mean': 0.0,
            'lightness_std': 0.0,
            'color_uniformity': 0.0
        }

def save_analysis_visualizations(image, output_dir):
    """Save multiple analysis visualizations"""
    visualizations = {}
    
    # 1. Red channel histogram
    hist_path = os.path.join(output_dir, "red_histogram.png")
    save_red_channel_histogram_fixed(image, hist_path)
    visualizations['histogram'] = hist_path
    
    # 2. Edge detection visualization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')
    
    edge_path = os.path.join(output_dir, "edge_analysis.png")
    plt.tight_layout()
    plt.savefig(edge_path, dpi=150, bbox_inches='tight')
    plt.close()
    visualizations['edges'] = edge_path
    
    # 3. Color channel analysis
    plt.figure(figsize=(15, 4))
    channels = ['Blue', 'Green', 'Red']
    colors = ['blue', 'green', 'red']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(image[:, :, i].flatten(), bins=50, color=colors[i], alpha=0.7)
        plt.title(f'{channels[i]} Channel Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    color_path = os.path.join(output_dir, "color_analysis.png")
    plt.tight_layout()
    plt.savefig(color_path, dpi=150, bbox_inches='tight')
    plt.close()
    visualizations['colors'] = color_path
    
    return visualizations

class TurbidityApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Turbidity Analyzer")
        master.geometry("450x400")
        master.resizable(False, False)

        self.image_path = None
        self.result_data = None

        # Main title
        tk.Label(master, text="Advanced Water Turbidity Analyzer", 
                font=("Arial", 16, "bold")).pack(pady=10)
        
        # Upload button
        tk.Button(master, text="Upload Image", command=self.upload_image,
                 bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=5)
        
        # Result display
        self.result_label = tk.Label(master, text="No image analyzed", 
                                   fg="gray", font=("Arial", 10))
        self.result_label.pack(pady=10)
        
        # Additional info label
        self.info_label = tk.Label(master, text="", fg="blue", font=("Arial", 9))
        self.info_label.pack(pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Generate PDF Report", 
                 command=self.generate_pdf, bg="#2196F3", fg="white",
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Save Visualizations", 
                 command=self.save_visualizations, bg="#FF9800", fg="white",
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        self.image_path = file_path  # Store the image path

        try:
            # Original analysis with error handling
            preprocessed, original = preprocess_image(file_path)
            avg_intensity, var, min_intensity, max_intensity = compute_intensity_metrics(preprocessed)
            edge_density = compute_edge_density(preprocessed)
            turbidity_index = compute_turbidity_index(min_intensity, max_intensity)
            ntu, m_red, turb_val = calculate_turbidity_from_red_channel(original)
            
            # Enhanced analysis with individual error handling
            try:
                particle_stats = analyze_particle_distribution(original)
            except Exception as e:
                print(f"Particle analysis failed: {e}")
                particle_stats = {
                    'particle_count': 0,
                    'avg_particle_size': 0.0,
                    'particle_density': 0.0,
                    'particle_uniformity': 1.0
                }
            
            try:
                texture_features = compute_texture_features(original)
            except Exception as e:
                print(f"Texture analysis failed: {e}")
                texture_features = {
                    'lbp_variance': 0.0,
                    'texture_energy': 0.0,
                    'texture_contrast': 0.0
                }
            
            try:
                color_stats = analyze_color_distribution(original)
            except Exception as e:
                print(f"Color analysis failed: {e}")
                color_stats = {
                    'blue_mean': 0.0,
                    'green_mean': 0.0,
                    'red_mean': 0.0,
                    'hue_std': 0.0,
                    'saturation_mean': 0.0,
                    'lightness_std': 0.0,
                    'color_uniformity': 0.0
                }

            self.result_data = {
                'image_name': os.path.basename(file_path),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ntu': ntu,
                'avg_intensity': round(float(avg_intensity), 2),
                'intensity_variance': round(float(var), 2),
                'turbidity_index': round(float(turbidity_index), 4),
                'edge_density': round(float(edge_density), 4),
                'm_red': round(float(m_red), 2),
                'turb_val': round(float(turb_val), 2),
                **particle_stats,
                **texture_features,
                **color_stats
            }

            # Update display
            self.result_label.config(
                text=f"NTU: {ntu} | Particles: {particle_stats['particle_count']} | Texture Energy: {texture_features['texture_energy']:.3f}",
                fg="black"
            )
            
            self.info_label.config(
                text=f"Red Mean: {round(float(m_red), 2)} | Color Uniformity: {color_stats['color_uniformity']:.3f}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}\n\nTrying with basic analysis only...")
            # Fallback to basic analysis only
            try:
                preprocessed, original = preprocess_image(file_path)
                ntu, m_red, turb_val = calculate_turbidity_from_red_channel(original)
                
                self.result_data = {
                    'image_name': os.path.basename(file_path),
                    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'ntu': ntu,
                    'm_red': round(float(m_red), 2),
                    'turb_val': round(float(turb_val), 2),
                    # Default values for missing features
                    'avg_intensity': 0.0,
                    'intensity_variance': 0.0,
                    'turbidity_index': 0.0,
                    'edge_density': 0.0,
                    'particle_count': 0,
                    'avg_particle_size': 0.0,
                    'particle_density': 0.0,
                    'particle_uniformity': 1.0,
                    'lbp_variance': 0.0,
                    'texture_energy': 0.0,
                    'texture_contrast': 0.0,
                    'blue_mean': 0.0,
                    'green_mean': 0.0,
                    'red_mean': 0.0,
                    'hue_std': 0.0,
                    'saturation_mean': 0.0,
                    'lightness_std': 0.0,
                    'color_uniformity': 0.0
                }
                
                self.result_label.config(
                    text=f"NTU: {ntu} (Basic analysis only)",
                    fg="orange"
                )
                
                messagebox.showinfo("Info", "Basic turbidity analysis completed successfully!")
                
            except Exception as e2:
                messagebox.showerror("Error", f"Complete analysis failed: {str(e2)}")
                return

    def generate_pdf(self):
        if not self.result_data:
            messagebox.showwarning("Warning", "Please analyze an image first.")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".pdf", 
            filetypes=[("PDF file", "*.pdf")]
        )
        if not output_path:
            return

        try:
            # Create temporary directory for visualizations
            temp_dir = tempfile.mkdtemp()
            
            # Load and create visualizations
            image = cv2.imread(self.image_path)
            visualizations = save_analysis_visualizations(image, temp_dir)
            
            # Create PDF
            c = canvas.Canvas(output_path)
            
            # Page 1: Main Results
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 800, "Advanced Turbidity Analysis Report")
            
            c.setFont("Helvetica", 10)
            c.drawString(50, 780, f"Generated: {self.result_data['datetime']}")
            c.drawString(50, 765, f"Image: {self.result_data['image_name']}")
            
            # Main turbidity result
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 735, f"Turbidity (NTU): {self.result_data['ntu']}")
            
            # Turbidity level indicator
            ntu = self.result_data['ntu']
            c.setFont("Helvetica", 10)
            c.drawString(50, 715, "Turbidity Level:")
            
            if ntu < 25:
                c.setFillColor(green)
                level_text = "Low (Clear water)"
            elif ntu < 100:
                c.setFillColor(yellow)
                level_text = "Medium (Slightly turbid)"
            else:
                c.setFillColor(red)
                level_text = "High (Very turbid)"
            
            bar_width = min(ntu * 2, 300)
            c.rect(150, 710, bar_width, 15, fill=True)
            c.setFillColor(green if ntu < 25 else yellow if ntu < 100 else red)
            c.drawString(50, 695, level_text)
            c.setFillColor('black')
            
            # Detailed results
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, 665, "Detailed Analysis Results:")
            
            c.setFont("Helvetica", 9)
            y_pos = 645
            
            # Basic measurements
            basic_results = [
                f"Average Intensity: {self.result_data['avg_intensity']}",
                f"Intensity Variance: {self.result_data['intensity_variance']}",
                f"Edge Density: {self.result_data['edge_density']}",
                f"Turbidity Index: {self.result_data['turbidity_index']}",
                f"Red Channel Mean: {self.result_data['m_red']}",
            ]
            
            for result in basic_results:
                c.drawString(50, y_pos, result)
                y_pos -= 15
            
            # Particle analysis
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y_pos - 10, "Particle Analysis:")
            c.setFont("Helvetica", 9)
            y_pos -= 25
            
            particle_results = [
                f"Particle Count: {self.result_data['particle_count']}",
                f"Average Particle Size: {self.result_data['avg_particle_size']} pixels",
                f"Particle Density: {self.result_data['particle_density']} per M pixels",
                f"Particle Uniformity: {self.result_data['particle_uniformity']}",
            ]
            
            for result in particle_results:
                c.drawString(50, y_pos, result)
                y_pos -= 15
            
            # Texture analysis
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y_pos - 10, "Texture Analysis:")
            c.setFont("Helvetica", 9)
            y_pos -= 25
            
            texture_results = [
                f"LBP Variance: {self.result_data['lbp_variance']}",
                f"Texture Energy: {self.result_data['texture_energy']}",
                f"Texture Contrast: {self.result_data['texture_contrast']}",
            ]
            
            for result in texture_results:
                c.drawString(50, y_pos, result)
                y_pos -= 15
            
            # Color analysis
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y_pos - 10, "Color Analysis:")
            c.setFont("Helvetica", 9)
            y_pos -= 25
            
            color_results = [
                f"RGB Means: R={self.result_data['red_mean']}, G={self.result_data['green_mean']}, B={self.result_data['blue_mean']}",
                f"Saturation Mean: {self.result_data['saturation_mean']}",
                f"Color Uniformity: {self.result_data['color_uniformity']}",
            ]
            
            for result in color_results:
                c.drawString(50, y_pos, result)
                y_pos -= 15
            
            # Start new page for visualizations
            c.showPage()
            
            # Page 2: Red Channel Histogram
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 800, "Red Channel Histogram Analysis")
            
            if 'histogram' in visualizations:
                c.drawImage(ImageReader(visualizations['histogram']), 50, 550, width=500, height=200)
            
            # Page 3: Edge Analysis
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 800, "Edge Detection Analysis")
            
            if 'edges' in visualizations:
                c.drawImage(ImageReader(visualizations['edges']), 50, 400, width=500, height=300)
            
            # Page 4: Color Channel Analysis
            c.showPage()
            c.setFont("Helvetica-Bold", 14) 
            c.drawString(50, 800, "Color Channel Distribution Analysis")
            
            if 'colors' in visualizations:
                c.drawImage(ImageReader(visualizations['colors']), 50, 400, width=500, height=300)
            
            c.save()
            
            # Clean up temporary files
            for viz_path in visualizations.values():
                if os.path.exists(viz_path):
                    os.remove(viz_path)
            os.rmdir(temp_dir)

            messagebox.showinfo("Success", "Comprehensive PDF report generated successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF: {str(e)}")

    def save_visualizations(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please analyze an image first.")
            return
        
        output_dir = filedialog.askdirectory()
        if not output_dir:
            return
        
        try:
            image = cv2.imread(self.image_path)
            visualizations = save_analysis_visualizations(image, output_dir)
            
            messagebox.showinfo("Success", 
                f"Visualizations saved to:\n" + 
                "\n".join([f"- {os.path.basename(path)}" for path in visualizations.values()])
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save visualizations: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TurbidityApp(root)
    root.mainloop()