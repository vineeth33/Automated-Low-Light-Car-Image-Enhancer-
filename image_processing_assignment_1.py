# -*- coding: utf-8 -*-
"""Image Processing Assignment-1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IEf66wzgd-ryTb4yIfVY7IQTDbziaUCR
"""

# @title Import Necessary Modules

# For opening and modifying images
import cv2

# For processing images as arrays
import numpy as np

# To display the images
from matplotlib import pyplot as plt

# For opening directory structures and manipulating files
import os

# @title Mount the drive folder

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# @title Define necessary functions for image processing

# Convert color image to grayscale image
def to_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate noise standard deviation using the Laplacian of the image.
def estimate_noise(image):
    gray_image = to_gray_image(image)
    noise_std = np.std(cv2.Laplacian(gray_image, cv2.CV_64F))
    return noise_std

# Calculate the mean pixel value in the grayscale image to assess brightness.
def calculate_brightness(image):
    gray_image = to_gray_image(image)
    return np.mean(gray_image)

# Use Canny edge detection to find edges and calculate their density.
def edge_density(image):
    gray_image = to_gray_image(image)
    edges = cv2.Canny(gray_image, 100, 200)
    return np.sum(edges > 0) / edges.size

def bilateral_filter(image):
    noise_level = estimate_noise(image)
    if noise_level > 20:
        d = 9  # Diameter of pixel neighborhood.
        sigmaColor = min(150, noise_level * 3)  # Filter intensity based on noise.
        sigmaSpace = min(150, noise_level * 3)  # Filter spatial extent based on noise.
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return image

def gamma_correction(image):
    brightness = calculate_brightness(image)
    if brightness < 100:  # Dark image
        gamma = 2.0
    elif brightness > 180:  # Bright image
        gamma = 0.8
    else:
        gamma = 1.2  # Slightly adjust for medium brightness
    return np.array(255 * (image / 255) ** (1 / gamma), dtype='uint8')

def clahe(image):
    brightness = calculate_brightness(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab_image)
    clahe_clip_limit = 3.0 if brightness < 100 else 2.0
    tileGridSize = (8, 8) if brightness < 100 else (10, 10)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def sharpening(image):
    edge_density_value = edge_density(image)
    if edge_density_value > 0.05:  # Strong edges
        alpha = 1.0
        beta = -0.5
    else:  # Weaker edges
        alpha = 1.5
        beta = -0.7
    gaussian_blurred = cv2.GaussianBlur(image, (9, 9), 10)
    return cv2.addWeighted(image, alpha, gaussian_blurred, beta, 0)

def image_processing(image):
    denoised_image = bilateral_filter(image)
    gamma_corrected_image = gamma_correction(denoised_image)
    clahe_image = clahe(gamma_corrected_image)
    final_image = sharpening(clahe_image)
    return final_image

# @title Call the image processing techniques on all images in the target drive folder

# Path to the drive folder containing the images
folder_path = '/content/drive/My Drive/IPA'

# List all images in the folder
image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image {image_file}")
        continue

    processed_image = image_processing(image)

    # Display original and processed images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image')
    plt.axis('off')
    plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Helper function to display images
def display_image(image, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Convert color image to grayscale image
def to_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate noise standard deviation using the Laplacian of the image.
def estimate_noise(image):
    gray_image = to_gray_image(image)
    noise_std = np.std(cv2.Laplacian(gray_image, cv2.CV_64F))
    return noise_std

# Calculate the mean pixel value in the grayscale image to assess brightness.
def calculate_brightness(image):
    gray_image = to_gray_image(image)
    return np.mean(gray_image)

# Use Canny edge detection to find edges and calculate their density.
def edge_density(image):
    gray_image = to_gray_image(image)
    edges = cv2.Canny(gray_image, 100, 200)
    return np.sum(edges > 0) / edges.size

# Apply Bilateral Filter
def bilateral_filter(image):
    noise_level = estimate_noise(image)
    if noise_level > 20:
        d = 9  # Diameter of pixel neighborhood.
        sigmaColor = min(150, noise_level * 3)  # Filter intensity based on noise.
        sigmaSpace = min(150, noise_level * 3)  # Filter spatial extent based on noise.
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return image

# Apply Gamma Correction
def gamma_correction(image):
    brightness = calculate_brightness(image)
    if brightness < 100:  # Dark image
        gamma = 2.0
    elif brightness > 180:  # Bright image
        gamma = 0.8
    else:
        gamma = 1.2  # Slightly adjust for medium brightness
    return np.array(255 * (image / 255) ** (1 / gamma), dtype='uint8')

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe(image):
    brightness = calculate_brightness(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab_image)
    clahe_clip_limit = 3.0 if brightness < 100 else 2.0
    tileGridSize = (8, 8) if brightness < 100 else (10, 10)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Apply Sharpening
def sharpening(image):
    edge_density_value = edge_density(image)
    if edge_density_value > 0.05:  # Strong edges
        alpha = 1.0
        beta = -0.5
    else:  # Weaker edges
        alpha = 1.5
        beta = -0.7
    gaussian_blurred = cv2.GaussianBlur(image, (9, 9), 10)
    return cv2.addWeighted(image, alpha, gaussian_blurred, beta, 0)

# Apply full image processing pipeline with intermediate visualizations
def image_processing_pipeline(image):
    # Step 1: Original Image
    display_image(image, "Step 1: Original Image")

    # Step 2: Convert to Grayscale
    gray_image = to_gray_image(image)
    display_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Step 2: Grayscale Image")

    # Step 3: Bilateral Filter (Denoising)
    denoised_image = bilateral_filter(image)
    display_image(denoised_image, "Step 3: Bilateral Filter (Denoising)")

    # Step 4: Gamma Correction
    gamma_corrected_image = gamma_correction(denoised_image)
    display_image(gamma_corrected_image, "Step 4: Gamma Correction")

    # Step 5: CLAHE (Contrast Enhancement)
    clahe_image = clahe(gamma_corrected_image)
    display_image(clahe_image, "Step 5: CLAHE (Contrast Enhancement)")

    # Step 6: Edge Detection
    edges = cv2.Canny(to_gray_image(clahe_image), 100, 200)
    edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    display_image(edge_image, "Step 6: Edge Detection")

    # Step 7: Sharpening
    final_image = sharpening(clahe_image)
    display_image(final_image, "Step 7: Sharpening (Final Output)")

    return final_image

# Process one image
# Specify the path to the image
image_path = '/content/drive/My Drive/image_processing/IPA/2015_02538.jpg'

# Load the image
image = cv2.imread(image_path)

if image is None:
    print(f"Error loading image {image_path}")
else:
    # Apply the image processing pipeline with step-by-step visualizations
    processed_image = image_processing_pipeline(image)

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# Convert color image to grayscale image
def to_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate noise standard deviation using the Laplacian of the image.
def estimate_noise(image):
    gray_image = to_gray_image(image)
    noise_std = np.std(cv2.Laplacian(gray_image, cv2.CV_64F))
    return noise_std

# Calculate the mean pixel value in the grayscale image to assess brightness.
def calculate_brightness(image):
    gray_image = to_gray_image(image)
    return np.mean(gray_image)

# Use Canny edge detection to find edges and calculate their density.
def edge_density(image):
    gray_image = to_gray_image(image)
    edges = cv2.Canny(gray_image, 100, 200)
    return np.sum(edges > 0) / edges.size

def bilateral_filter(image):
    noise_level = estimate_noise(image)
    if noise_level > 20:
        d = 9  # Diameter of pixel neighborhood.
        sigmaColor = min(150, noise_level * 3)  # Filter intensity based on noise.
        sigmaSpace = min(150, noise_level * 3)  # Filter spatial extent based on noise.
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return image

def gamma_correction(image):
    brightness = calculate_brightness(image)
    if brightness < 100:  # Dark image
        gamma = 2.0
    elif brightness > 180:  # Bright image
        gamma = 0.8
    else:
        gamma = 1.2  # Slightly adjust for medium brightness
    return np.array(255 * (image / 255) ** (1 / gamma), dtype='uint8')

def clahe(image):
    brightness = calculate_brightness(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab_image)
    clahe_clip_limit = 3.0 if brightness < 100 else 2.0
    tileGridSize = (8, 8) if brightness < 100 else (10, 10)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def sharpening(image):
    edge_density_value = edge_density(image)
    if edge_density_value > 0.05:  # Strong edges
        alpha = 1.0
        beta = -0.5
    else:  # Weaker edges
        alpha = 1.5
        beta = -0.7
    gaussian_blurred = cv2.GaussianBlur(image, (9, 9), 10)
    return cv2.addWeighted(image, alpha, gaussian_blurred, beta, 0)

# Image processing pipeline
def image_processing(image):
    denoised_image = bilateral_filter(image)
    gamma_corrected_image = gamma_correction(denoised_image)
    clahe_image = clahe(gamma_corrected_image)
    final_image = sharpening(clahe_image)
    return final_image

# Function to calculate PSNR
def calculate_psnr(original_image, processed_image):
    mse = np.mean((original_image - processed_image) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match, so infinite PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Function to calculate SSIM
def calculate_ssim(original_image, processed_image):
    gray_original = to_gray_image(original_image)
    gray_processed = to_gray_image(processed_image)
    ssim_value, _ = ssim(gray_original, gray_processed, full=True)
    return ssim_value

# Process the image and compare PSNR and SSIM
def compare_images(original_image, processed_image):
    psnr_value = calculate_psnr(original_image, processed_image)
    ssim_value = calculate_ssim(original_image, processed_image)
    return psnr_value, ssim_value

# Function to process all images in a folder
def process_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    results = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        original_image = cv2.imread(image_path)
        processed_image = image_processing(original_image)

        # Calculate PSNR and SSIM
        psnr_value, ssim_value = compare_images(original_image, processed_image)
        results.append((image_file, psnr_value, ssim_value))

        # Optionally save the processed image
        processed_image_path = os.path.join(folder_path, 'processed_' + image_file)
        cv2.imwrite(processed_image_path, processed_image)

    return results

# Example usage:
folder_path = '/content/drive/My Drive/image_processing/IPA'  # Replace with your folder path
results = process_folder(folder_path)

# Print results for each image
for result in results:
    print(f"Image: {result[0]}, PSNR: {result[1]}, SSIM: {result[2]}")

!pip install easyocr

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr

# Initialize YOLOv5 model for car and license plate detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use a pre-trained YOLOv5 small model
reader = easyocr.Reader(['en'])

# Helper function to display images
def display_image(image, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Convert color image to grayscale image
def to_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate noise standard deviation using the Laplacian of the image.
def estimate_noise(image):
    gray_image = to_gray_image(image)
    noise_std = np.std(cv2.Laplacian(gray_image, cv2.CV_64F))
    return noise_std

# Calculate the mean pixel value in the grayscale image to assess brightness.
def calculate_brightness(image):
    gray_image = to_gray_image(image)
    return np.mean(gray_image)

# Use Canny edge detection to find edges and calculate their density.
def edge_density(image):
    gray_image = to_gray_image(image)
    edges = cv2.Canny(gray_image, 100, 200)
    return np.sum(edges > 0) / edges.size

# Apply Bilateral Filter
def bilateral_filter(image):
    noise_level = estimate_noise(image)
    if noise_level > 20:
        d = 9
        sigmaColor = min(150, noise_level * 3)
        sigmaSpace = min(150, noise_level * 3)
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return image

# Apply Gamma Correction
def gamma_correction(image):
    brightness = calculate_brightness(image)
    if brightness < 100:
        gamma = 2.0
    elif brightness > 180:
        gamma = 0.8
    else:
        gamma = 1.2
    return np.array(255 * (image / 255) ** (1 / gamma), dtype='uint8')

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe(image):
    brightness = calculate_brightness(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab_image)
    clahe_clip_limit = 3.0 if brightness < 100 else 2.0
    tileGridSize = (8, 8) if brightness < 100 else (10, 10)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Apply Sharpening
def sharpening(image):
    edge_density_value = edge_density(image)
    if edge_density_value > 0.05:
        alpha = 1.0
        beta = -0.5
    else:
        alpha = 1.5
        beta = -0.7
    gaussian_blurred = cv2.GaussianBlur(image, (9, 9), 10)
    return cv2.addWeighted(image, alpha, gaussian_blurred, beta, 0)

# Image processing pipeline
def image_processing_pipeline(image):
    display_image(image, "Step 1: Original Image")
    gray_image = to_gray_image(image)
    display_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Step 2: Grayscale Image")
    denoised_image = bilateral_filter(image)
    display_image(denoised_image, "Step 3: Bilateral Filter (Denoising)")
    gamma_corrected_image = gamma_correction(denoised_image)
    display_image(gamma_corrected_image, "Step 4: Gamma Correction")
    clahe_image = clahe(gamma_corrected_image)
    display_image(clahe_image, "Step 5: CLAHE (Contrast Enhancement)")
    edges = cv2.Canny(to_gray_image(clahe_image), 100, 200)
    edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    display_image(edge_image, "Step 6: Edge Detection")
    final_image = sharpening(clahe_image)
    display_image(final_image, "Step 7: Sharpening (Final Output)")
    return final_image

# Detect car and license plate
def detect_car_and_license_plate(image):
    results = model(image)
    detections = results.pandas().xyxy[0]
    cars = detections[detections['name'] == 'car']
    license_plates = detections[detections['name'] == 'license plate']

    for _, car in cars.iterrows():
        xmin, ymin, xmax, ymax = map(int, [car['xmin'], car['ymin'], car['xmax'], car['ymax']])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    license_texts = []
    for _, plate in license_plates.iterrows():
        xmin, ymin, xmax, ymax = map(int, [plate['xmin'], plate['ymin'], plate['xmax'], plate['ymax']])
        license_plate_img = image[ymin:ymax, xmin:xmax]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ocr_result = reader.readtext(license_plate_img)
        if ocr_result:
            text = ocr_result[0][-2]
            license_texts.append(text)
            print(f"Recognized License Plate Text: {text}")
    display_image(image, "Detected Car and License Plate")
    return license_texts

# Process the image and detect objects
image_processing_folder = '/content/drive/My Drive/IPA/'
for filename in os.listdir(image_processing_folder):
    # Construct the full path to the image file
    image_path = os.path.join(image_processing_folder, filename)

    # Check if the file is an image based on extension
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image {image_path}")
            continue  # Skip to the next file if loading fails

        print(f"Processing {filename}...")

        # Apply the image processing pipeline
        processed_image = image_processing_pipeline(image)

        # Detect car and license plate in the processed image
        license_texts = detect_car_and_license_plate(processed_image)

        if license_texts:
            print(f"Detected license plate texts in {filename}: {license_texts}")
        else:
            print(f"No license plate detected in {filename}")

print("Processing complete for all images in the folder.")

!pip install pytesseract

# Import necessary libraries and install dependencies
import sys
!{sys.executable} -m pip install pytesseract
!apt-get install -y tesseract-ocr

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from sklearn.cluster import KMeans
import pytesseract
import os
from google.colab import drive
from google.colab.patches import cv2_imshow
from tqdm.notebook import tqdm

# Mount Google Drive
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully")
except:
    print("Error mounting Google Drive. Please check your permissions and try again.")
    sys.exit(1)

# Path for input folder on Google Drive
input_folder_path = '/content/drive/MyDrive/image_processing/IPA'  # Replace with your input folder path
if not os.path.exists(input_folder_path):
    print(f"Error: The folder {input_folder_path} does not exist. Please check the path and try again.")
    sys.exit(1)

# Load a pre-trained EfficientNetB0 model for object classification
model = EfficientNetB0(weights='imagenet', include_top=True)

def enhance_image(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel back with A and B channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Apply additional contrast enhancement
    alpha = 1.5  # Contrast control
    beta = 10    # Brightness control
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    return enhanced

def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = gray[topx:bottomx+1, topy:bottomy+1]

        plate_text = pytesseract.image_to_string(cropped, config='--psm 8')
        cv2.rectangle(image, (topy, topx), (bottomy, bottomx), (0, 255, 0), 3)
        cv2.putText(image, plate_text.strip(), (topy, topx - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        plate_text = "No plate detected"

    return image, plate_text.strip()

def detect_car_color(image):
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask to focus on the car body
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

    # Reshape the image to be a list of pixels
    pixels = masked_image.reshape(-1, 3)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[0]

    # Convert HSV to RGB
    dominant_color_rgb = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_HSV2RGB)[0][0]

    # Define color ranges and names
    color_ranges = {
        'Red': ([0, 100, 100], [10, 255, 255]),
        'Yellow': ([20, 100, 100], [30, 255, 255]),
        'Green': ([40, 100, 100], [70, 255, 255]),
        'Blue': ([100, 100, 100], [130, 255, 255]),
        'White': ([0, 0, 200], [180, 30, 255]),
        'Black': ([0, 0, 0], [180, 255, 30]),
        'Gray': ([0, 0, 70], [180, 30, 200])
    }

    # Determine the color name
    color_name = 'Unknown'
    for name, (lower, upper) in color_ranges.items():
        if np.all(dominant_color >= lower) and np.all(dominant_color <= upper):
            color_name = name
            break

    return color_name, dominant_color_rgb

def classify_objects(image):
    img = cv2.resize(image, (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

def extract_car_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area to focus on larger objects (likely to be cars)
    min_area = 1000  # Adjust this value based on your images
    car_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Extract features
    features = []
    for contour in car_contours:
        # Contour area
        area = cv2.contourArea(contour)

        # Contour perimeter
        perimeter = cv2.arcLength(contour, True)

        # Contour approximation
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        features.append({
            'area': area,
            'perimeter': perimeter,
            'corners': len(approx),
            'aspect_ratio': aspect_ratio
        })

    return features

def process_image(image_path):
    image = cv2.imread(image_path)
    enhanced_image = enhance_image(image)

    image_with_plate, plate_text = detect_number_plate(enhanced_image)
    car_color, rgb_color = detect_car_color(enhanced_image)
    object_predictions = classify_objects(enhanced_image)
    car_features = extract_car_features(enhanced_image)

    print(f"Number Plate: {plate_text}")
    print(f"Car Color: {car_color}")
    print("RGB Color:", rgb_color)
    print("Object Predictions:")
    for pred in object_predictions:
        print(f"  {pred[1]}: {pred[2]:.2f}")
    print("Car Features:")
    for i, feature in enumerate(car_features):
        print(f"  Object {i+1}:")
        for key, value in feature.items():
            print(f"    {key}: {value}")

    # Display the processed image
    cv2_imshow(image_with_plate)

# Process all images in the input folder
for filename in tqdm(os.listdir(input_folder_path), desc="Processing images"):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder_path, filename)
        print(f"\nProcessing: {filename}")
        process_image(image_path)

print("Processing complete.")

# Install required packages
!pip install pytesseract

# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import pytesseract
import os
from google.colab import drive
from google.colab.patches import cv2_imshow
from tqdm.notebook import tqdm

# Mount Google Drive
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully")
except:
    print("Error mounting Google Drive. Please check your permissions and try again.")
    import sys
    sys.exit(1)

# Path for input folder on Google Drive
input_folder_path = '/content/drive/MyDrive/image_processing/IPA'  # Replace with your input folder path
if not os.path.exists(input_folder_path):
    print(f"Error: The folder {input_folder_path} does not exist. Please check the path and try again.")
    import sys
    sys.exit(1)

# Load pre-trained model
model = MobileNetV2(weights='imagenet', include_top=True)

def enhance_image(image):
    # Convert to LAB color space and apply CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

def detect_and_classify_objects(image):
    height, width = image.shape[:2]
    img = cv2.resize(image, (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    # Filter for vehicle-related predictions
    vehicle_predictions = [pred for pred in decoded_predictions if pred[1] in ['car', 'truck', 'bus', 'motorcycle']]

    if vehicle_predictions:
        # Use the highest confidence vehicle prediction
        best_pred = vehicle_predictions[0]
        return [(0, 0, width, height)], [best_pred]
    else:
        return [], []

def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio
    possible_plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:
                possible_plates.append((x, y, w, h))

    # If plates found, use the largest one
    if possible_plates:
        x, y, w, h = max(possible_plates, key=lambda b: b[2] * b[3])
        plate_img = gray[y:y+h, x:x+w]
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 8')
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, plate_text.strip(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        plate_text = "No plate detected"

    return image, plate_text.strip()

def process_image(image_path):
    image = cv2.imread(image_path)

    enhanced_image = enhance_image(image)

    cars, predictions = detect_and_classify_objects(enhanced_image)

    results = []
    for (x, y, w, h), prediction in zip(cars, predictions):
        car_image = enhanced_image[y:y+h, x:x+w]

        image_with_plate, plate_text = detect_number_plate(car_image)

        results.append({
            'box': (x, y, w, h),
            'plate': plate_text,
            'prediction': prediction
        })

    # Draw results on the image
    for result in results:
        x, y, w, h = result['box']
        cv2.rectangle(enhanced_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(enhanced_image, f"Plate: {result['plate']}",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the processed image
    cv2_imshow(enhanced_image)

    return results

# Process all images in the input folder
for filename in tqdm(os.listdir(input_folder_path), desc="Processing images"):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder_path, filename)
        print(f"\nProcessing: {filename}")
        results = process_image(image_path)
        for i, result in enumerate(results):
            print(f"Vehicle {i+1}:")
            print(f"  Number Plate: {result['plate']}")
            print(f"  Prediction: {result['prediction'][1]} (confidence: {result['prediction'][2]:.2f})")
        print("\n")

print("Processing complete.")

# Install required packages
!pip install ultralytics opencv-python-headless scikit-learn

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from google.colab import drive
from google.colab.patches import cv2_imshow
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set up the input folder path
input_folder_path = '/content/drive/MyDrive/IPA'  # Replace with your folder path

# Load the YOLO model
model = YOLO('yolov8n.pt')

def detect_cars(image):
    results = model(image)
    cars = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.cls == 2:  # Class 2 is 'car' in COCO dataset
                x1, y1, x2, y2 = box.xyxy[0]
                cars.append((int(x1), int(y1), int(x2), int(y2)))
    return cars

def extract_dominant_colors(image, n_colors=5):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)

    # Get the colors
    colors = kmeans.cluster_centers_

    # Get the labels
    labels = kmeans.labels_

    # Count the occurrences of each label
    counts = np.bincount(labels)

    # Sort colors by count
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_percentages = counts[sorted_indices] / len(labels)

    return sorted_colors, sorted_percentages

def get_color_name(rgb):
    color_ranges = {
        'Red': ([0, 100, 100], [10, 255, 255]),
        'Yellow': ([20, 100, 100], [30, 255, 255]),
        'Green': ([40, 100, 100], [70, 255, 255]),
        'Blue': ([100, 100, 100], [130, 255, 255]),
        'White': ([0, 0, 200], [180, 30, 255]),
        'Black': ([0, 0, 0], [180, 255, 30]),
        'Gray': ([0, 0, 70], [180, 30, 200])
    }

    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    for name, (lower, upper) in color_ranges.items():
        if np.all(hsv >= lower) and np.all(hsv <= upper):
            return name
    return 'Unknown'

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect cars
    cars = detect_cars(image)

    # Process each detected car
    for i, (x1, y1, x2, y2) in enumerate(cars):
        car_image = image[y1:y2, x1:x2]

        # Extract dominant colors
        colors, percentages = extract_dominant_colors(car_image)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Print results
        print(f"Car {i+1}:")
        for j, (color, percentage) in enumerate(zip(colors, percentages)):
            color_name = get_color_name(color)
            print(f"  Color {j+1}: {color_name} - {percentage*100:.2f}%")
        print()

    # Display the image
    cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Process all images in the input folder
for filename in os.listdir(input_folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder_path, filename)
        print(f"Processing: {filename}")
        process_image(image_path)
        print("\n" + "="*50 + "\n")

print("Processing complete.")