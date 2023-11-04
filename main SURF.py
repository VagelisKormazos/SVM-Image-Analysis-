import os
import cv2
import joblib
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
# Assuming you've added these import statements
import matplotlib

matplotlib.use('TkAgg')  # You can try 'Qt5Agg' or 'Agg' as well

# Function to compute superpixel LAB data
def compute_superpixel_lab_data(image_lab, segments, segment_id):
    mask = (segments == segment_id)
    segment_pixels_lab = image_lab[mask]
    mean_lab_color = np.mean(segment_pixels_lab, axis=0)
    return {'mask': mask, 'lab_data': mean_lab_color}

def compute_surf_features(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Check the OpenCV version
    is_cv4 = cv2.__version__.startswith('4.')

    if is_cv4:
        surf = cv2.SIFT_create()  # SIFT is available in the features2d module
    else:
        surf = cv2.xfeatures2d.SURF_create()

    keypoints, descriptors = surf.detectAndCompute(gray, None)
    return descriptors


# Path to the folder with images
folder_path = r'C:\Users\Vagelis\Desktop\Dataset\All'

# Load all images from the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# Select the first 2 images for training purposes
image_files = image_files[:5]

# Apply the process for each image
superpixel_data_all = []

for image_file in image_files:
    # Load the image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute SURF features
    surf_features = compute_surf_features(image_rgb)

    # Convert the image to the Lab color space
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # Apply the SLIC algorithm to extract superpixels
    num_segments = 100
    segments = slic(image_rgb, n_segments=num_segments, compactness=10)

    # Extract unique values of superpixels
    unique_segments = np.unique(segments)

    # Extract LAB color information for superpixels using SURF features
    superpixel_data_all.extend([compute_superpixel_lab_data(image_lab, segments, segment_id)
                                for segment_id in unique_segments])

    # Save contours and Lab data in memory
    superpixel_data_all.extend([compute_superpixel_lab_data(image_lab, segments, segment_id) for segment_id in unique_segments])

# Create the train set and the test set
train_data, test_data = train_test_split(superpixel_data_all, test_size=0.2, random_state=42)

# Extract data and labels for the classifier
X_train = np.array([d['lab_data'][1:] for d in train_data])
Y_train = np.array([d['lab_data'][0] for d in train_data])

X_test = np.array([d['lab_data'][1:] for d in test_data])
Y_test = np.array([d['lab_data'][0] for d in test_data])

# Normalize labels to the range [0, 1]
Y_train_normalized = Y_train / 255.0
Y_test_normalized = Y_test / 255.0

# Train the LinearSVR
regressor = LinearSVR(max_iter=10000)
regressor.fit(X_train, Y_train_normalized)

# Predict on the test data
Y_pred = regressor.predict(X_test)

# Reverse the normalization to the Lab values range [0, 1]
Y_pred_denormalized = Y_pred * 255.0

# Display real values (Y_test) and predicted values (Y_pred)
print(f'Real Values (Y_test): {Y_test}')
print(f'Predicted Values (Y_pred): {Y_pred_denormalized}')

# Display the original image with superpixel boundaries
image_with_boundaries = image_rgb.copy()
for segment_id in unique_segments:
    boundary_mask = np.zeros_like(segments)
    boundary_mask[segments == segment_id] = 1
    contours, _ = cv2.findContours(boundary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_with_boundaries, contours, -1, (255, 0, 0), 2)

# Display the predicted coloring with superpixel boundaries
predicted_image_with_boundaries = image_rgb.copy()
for segment_id, lab_data in zip(unique_segments, Y_pred_denormalized):
    predicted_image_with_boundaries[segments == segment_id] = lab_data

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_with_boundaries)
plt.title('Original Image with Superpixel Boundaries')

plt.subplot(1, 2, 2)
plt.imshow(predicted_image_with_boundaries)
plt.title('Predicted Coloring with Superpixel Boundaries')

plt.show()

# Add a question to save the model
save_model = input("Do you want to save the model? (Y/NO): ").lower()

if save_model == 'y':
    # Save the model
    model_filename = 'linear_svr_model.sav'
    joblib.dump(regressor, model_filename)
    print(f'The model is saved in the file {model_filename}.')
else:
    print('The model was not saved.')
