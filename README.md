# 💧 Image-Based Water Turbidity Analysis System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Project Overview
The **Water Turbidity Analysis System** is a sophisticated Python-based application designed to estimate water turbidity from digital images. By employing advanced image processing techniques, it provides a cost-effective and non-invasive alternative to traditional turbidimeters.

The system utilizes a multi-metric approach, including:
- **Red Channel Analysis:** Utilizing a quadratic model to correlate red pixel intensity with NTU.
- **Edge Density Mapping:** Applying Canny Edge Detection to quantify suspended particles.
- **Statistical Intensity Metrics:** Analyzing pixel distribution (mean, variance, range) to characterize water clarity.
- **Particle Distribution:** Performing connected component analysis for micro-level particle density estimation.

## 🏷️ Tags
`Python`, `OpenCV`, `Tkinter`, `Image Processing`, `Water Quality`, `Turbidity Analysis`, `Computer Vision`, `Data Visualization`, `PDF Reporting`, `Environmental Monitoring`

---

## 📸 Sample GUI & Output
### User Interface
![GUI Sample](output/sample_tab.png)

### Analysis Results
| Feature | Visualization |
| :--- | :--- |
| **Original Sample** | ![Original Sample](output/sample_output.png) |
| **Red Channel Histogram** | ![Red Histogram](output/red_histogram.png) |
| **Edge Analysis** | ![Edge Analysis](output/edge_analysis.png) |

---

## 🧠 Core Features
- **Intuitive GUI:** Built with Tkinter for easy image uploading and real-time feedback.
- **Automated Preprocessing:** Includes watermark removal, noise reduction (Median Filter), and smoothing (Gaussian Blur).
- **Advanced Metrics:**
  - Average Intensity & Variance
  - Turbidity Index
  - Edge Density (Canny)
  - Particle Count & Density
  - Particle Uniformity
- **Visual Analytics:** Generates red channel histograms and edge maps.
- **Professional Reporting:** Automatically exports results into a detailed **PDF report** with color-coded NTU severity bars.

---

## 🔧 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
```bash
pip install opencv-python numpy matplotlib reportlab scikit-image scipy
```

### 3. Run the Application
```bash
python turbidity_gui.py
```

---

## 🛠️ Technologies Used
- **OpenCV:** Core image processing and feature extraction.
- **Tkinter:** Graphical User Interface.
- **Matplotlib:** Data visualization and histogram plotting.
- **ReportLab:** PDF report generation.
- **NumPy/SciPy/Scikit-Image:** Numerical computing and advanced image analysis.
