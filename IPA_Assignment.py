# For opening and modifying images
import cv2

# For processing images as arrays
import numpy as np

# To display the images
from matplotlib import pyplot as plt

# For opening directory structures and manipulating files
import os

#####################################################################################################
# Image Enhancement

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
        d = 5  # Diameter of pixel neighborhood.
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

def enhance_image(image):
    denoised_image = bilateral_filter(image)
    gamma_corrected_image = gamma_correction(denoised_image)
    clahe_image = clahe(gamma_corrected_image)
    final_image = sharpening(clahe_image)
    return final_image

#######################################################################################################################
# Image Detection

from ultralytics import YOLO
from sklearn.cluster import KMeans

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

def process_image(image):
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

    # Return the image
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#######################################################################################################################
# Execute the above methods over a dataset

# Path to the drive folder containing the images
folder_path = './IPA-Original-Images/'
enhanced_path = './IPA-Enhanced-Images/'
processed_path = './IPA-Processed-Images/'

# List all images in the folder
image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image {image_file}")
        continue

    enhanced_image = enhance_image(image)
    processed_image = process_image(enhanced_image)

    # Save the enhanced and processed images
    # cv2.imwrite(enhanced_path + image_file, enhanced_image)
    # cv2.imwrite(processed_path + image_file, processed_image)

    # Display original, enhanced, and processed images side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image' +'/'+image_file)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image')
    plt.axis('off')

    plt.show()
