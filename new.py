import cv2
import numpy as np

def detect_people():
    # Load pre-trained HOG detector for pedestrian detection
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Open camera
    camera = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        if ret:
            # Step 1: Edge Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Step 2: Segmentation (Optional, you can experiment with different segmentation techniques)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Step 3: Feature Extraction and Person Detection
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Adjust threshold as needed
                    x, y, w, h = cv2.boundingRect(contour)

                    # Step 4: Filtering and Classification (Here you can apply additional filters or classifiers)
                    # Use HOG detector to refine the detection
                    person, _ = HOGCV.detectMultiScale(frame[y:y + h, x:x + w], winStride=(4, 4), padding=(8, 8), scale=1.03)
                    for (px, py, pw, ph) in person:
                        cv2.rectangle(frame, (x + px, y + py), (x + px + pw, y + py + ph), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Human Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

# Call the function to detect people from live camera feed
detect_people()
