import os
import cv2
import joblib
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
import matplotlib
matplotlib.use('TkAgg')  # You can try 'Qt5Agg' or 'Agg' as well

# Συνάρτηση για τον υπολογισμό του μέσου χρώματος στον χώρο Lab για ένα υπερ-εικονοστοιχείο
def compute_superpixel_lab_data(image_lab, segments, segment_id):
    mask = (segments == segment_id)
    segment_pixels_lab = image_lab[mask]
    mean_lab_color = np.mean(segment_pixels_lab, axis=0)
    return {'mask': mask, 'lab_data': mean_lab_color}

# Διαδρομή προς τον φάκελο με τις εικόνες
folder_path = r'C:\Users\Vagelis\Desktop\Dataset\All'

# Φορτώστε όλες τις εικόνες από τον φάκελο
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# Επιλογή των πρώτων 2 εικόνων για τον εκπαιδευτικό σκοπό
image_files = image_files[:4]

# Εφαρμόστε τη διαδικασία για κάθε εικόνα
superpixel_data_all = []

for image_file in image_files:
    # Φορτώστε την εικόνα
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Μετατροπή της εικόνας στον χώρο Lab
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # Εφαρμόστε τον αλγόριθμο SLIC για να εξάγετε υπερ-εικονοστοιχεία
    num_segments = 500
    segments = slic(image_rgb, n_segments=num_segments, compactness=15)

    # Εξαγωγή μοναδικών τιμών των υπερ-εικονοστοιχείων
    unique_segments = np.unique(segments)

    # Αποθήκευση περιγραμμάτων και δεδομένων Lab στη μνήμη
    superpixel_data_all.extend([compute_superpixel_lab_data(image_lab, segments, segment_id) for segment_id in unique_segments])

# Δημιουργία του train set και του test set
train_data, test_data = train_test_split(superpixel_data_all, test_size=0.2, random_state=42)

# Εξαγωγή δεδομένων και ετικετών για τον ταξινομητή
X_train = np.array([d['lab_data'][1:] for d in train_data])
Y_train = np.array([d['lab_data'][0] for d in train_data])

X_test = np.array([d['lab_data'][1:] for d in test_data])
Y_test = np.array([d['lab_data'][0] for d in test_data])

# Κανονικοποίηση των ετικετών στο εύρος [0, 1]
Y_train_normalized = Y_train / 255.0
Y_test_normalized = Y_test / 255.0

# Εκπαίδευση του LinearSVR
regressor = LinearSVR(max_iter=10000)
regressor.fit(X_train, Y_train_normalized)

# Πρόβλεψη στα δεδομένα ελέγχου
Y_pred = regressor.predict(X_test)

# Αντιστροφή της κανονικοποίησης στο εύρος [0, 1] στις τιμές Lab
Y_pred_denormalized = Y_pred * 255.0

# Εμφάνιση των πραγματικών τιμών (Y_test) και των προβλεπόμενων τιμών (Y_pred)
print(f'Real Values (Y_test): {Y_test}')
print(f'Predicted Values (Y_pred): {Y_pred_denormalized}')

# Εμφάνιση της αρχικής εικόνας με τα όρια των υπερ-εικονοστοιχείων
image_with_boundaries = image_rgb.copy()
for segment_id in unique_segments:
    boundary_mask = np.zeros_like(segments)
    boundary_mask[segments == segment_id] = 1
    contours, _ = cv2.findContours(boundary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_with_boundaries, contours, -1, (255, 0, 0), 2)

# Εμφάνιση του προβλεπόμενου χρωματισμού με τα όρια των υπερ-εικονοστοιχείων
predicted_image_with_boundaries = image_rgb.copy()
for segment_id, lab_data in zip(unique_segments, Y_pred_denormalized):
    predicted_image_with_boundaries[segments == segment_id] = lab_data

# Εμφάνιση των αποτελεσμάτων
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_with_boundaries)
plt.title('Αρχική Εικόνα με Όρια Υπερ-εικονοστοιχείων')

plt.subplot(1, 2, 2)
plt.imshow(predicted_image_with_boundaries)
plt.title('Προβλεπόμενος Χρωματισμός με Όρια Υπερ-εικονοστοιχείων')

plt.show()

# Προσθήκη ερώτησης για αποθήκευση του μοντέλου
save_model = input("Θέλετε να αποθηκεύσετε το μοντέλο; (Y/NO): ").lower()

if save_model == 'Y':
    # Αποθήκευση του μοντέλου
    model_filename = 'linear_svr_model.sav'
    joblib.dump(regressor, model_filename)
    print(f'Το μοντέλο αποθηκεύτηκε στο αρχείο {model_filename}.')
else:
    print('Το μοντέλο δεν αποθηκεύτηκε.')