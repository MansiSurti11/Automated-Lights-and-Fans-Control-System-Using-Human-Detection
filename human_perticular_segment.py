import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera, change to another number if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get frame dimensions
    height, width, _ = frame.shape

    # Define coordinates for segmenting the frame into 6 parts
    x_segments = [0, width // 3, 2 * (width // 3), width]
    y_segments = [0, height // 2, height]

    # Segment the frame into 6 parts
    segments = []
    for i in range(len(y_segments) - 1):
        for j in range(len(x_segments) - 1):
            segment = frame[y_segments[i]:y_segments[i+1], x_segments[j]:x_segments[j+1]]
            segments.append(segment)

    # Choose a segment (e.g., segment 1) for skin color detection
    segment_of_interest = segments[0]

    # Convert segment to HSV color space
    hsv_segment = cv2.cvtColor(segment_of_interest, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask for skin color
    mask_skin = cv2.inRange(hsv_segment, lower_skin, upper_skin)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours (potential humans) are present in the segment
    if len(contours) > 0:
        print("A human is present in the segment.")
    else:
        print("No human detected in the segment.")

    # Display segmented parts
    for i, segment in enumerate(segments):
        cv2.imshow(f'Segment {i+1}', segment)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
