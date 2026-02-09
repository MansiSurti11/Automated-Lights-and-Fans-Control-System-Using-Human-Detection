# Computer Vision and Audio Processing Project

This project contains a collection of Python scripts for various computer vision tasks (such as human detection, image segmentation, and background subtraction) and audio processing (voice-to-text using Google Cloud Speech-to-Text).

## Project Overview

- **Human/Pedestrian Detection:** Uses HOG (Histogram of Oriented Gradients) and OpenCV's pre-trained SVM detectors to identify people in live camera feeds.
- **Image Segmentation:** Divides the camera feed into multiple segments for targeted analysis.
- **Background Subtraction:** Implements techniques to isolate moving objects from a static background.
- **Feature Extraction:** Extracts HOG features and visualizes them.
- **Audio to Text:** Transcribes audio files using the Google Cloud Speech-to-Text API.

## Prerequisites

- Python 3.x
- A working camera (for vision scripts)
- (Optional) Google Cloud Service Account credentials (for `at2.py`)

## Installation

1. **Clone the repository (or navigate to the project directory):**
   ```bash
   cd "Minor Project-1"
   ```

2. **Install the required dependencies:**
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running the Scripts

Below are the steps to run the various modules in this project:

### 1. Human Detection
To run the main human detection script:
```bash
python new.py
```
*Note: Press 'q' to exit the camera view.*

### 2. Image Segmentation
To see the live camera feed segmented into 6 parts:
```bash
python image_segmentatio.py
```

### 3. Pedestrian Detection with HOG Visualisation
To see pedestrian detection with HOG feature visualization:
```bash
python feature_extraction.py
```

### 4. Audio to Text Transcription
*Note: This script requires a Google Cloud Service Account JSON key.*
```bash
python at2.py
```
Make sure to update the `file_path` in `at2.py` with your audio file's path.

### 5. Other Scripts
- `background_subtraction.py`: Run for basic background subtraction demo.
- `human_perticular_segment.py`: Detects skin color in specific segments of the camera feed.

## Dependencies

- `opencv-python`: For image and video processing.
- `numpy`: For numerical computations.
- `scikit-image`: For advanced image processing algorithms.
- `matplotlib`: For visualization (used in feature extraction).
- `google-cloud-speech`: For speech-to-text functionality.
