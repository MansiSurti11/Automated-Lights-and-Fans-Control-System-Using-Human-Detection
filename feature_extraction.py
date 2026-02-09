import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import cv2

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera, change to another number if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate HOG features and visualization
    fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')

    # Rescale HOG features for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Threshold the image to find edges
    _, binary = cv2.threshold(hog_image_rescaled, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected pedestrians
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display result
    cv2.imshow('Pedestrian Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
