import cv2

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

    # Display segmented parts
    for i, segment in enumerate(segments):
        cv2.imshow(f'Segment {i+1}', segment)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
