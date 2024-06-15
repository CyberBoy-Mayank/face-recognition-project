---

# face-recognition-project

This project implements a real-time face recognition system using Python. The system captures images, trains a model with the captured faces, and then recognizes and displays names near the recognized faces in a live video feed. It leverages OpenCV for video capture and face detection, and uses a machine learning model for face recognition.

# Features

- **Image Capture**: Captures images from a in-built webcam for training purposes.
- **Model Training**: Trains a face recognition model using the captured images.
- **Real-Time Recognition**: Recognizes faces in a live video feed and displays names near the recognized faces.

# Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/CyberBoy-Mayank/face-recognition-project.git
    cd face-recognition-project
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

# Usage

1. Run the main script:
    ```sh
    python main.py
    ```

# Example Workflow

1. **Capture Images**: Use the `capture_images.py` script to capture images from your webcam and label them with names.
2. **Train Model**: Run the `train_model.py` script to train the face recognition model using the captured images.
3. **Recognize Faces**: Use the `recognize_faces.py` script to start the webcam feed and recognize faces in real-time, displaying names near the recognized faces.

# Benefits

- **High Accuracy**: Utilizes a robust face recognition model for accurate identification.
- **Real-Time Processing**: Efficient algorithms ensure real-time performance for live applications.
- **Easy to Use**: Simple commands for capturing images, training the model, and recognizing faces.
- **Customizable**: Easily extendable to include additional features or integrate with other systems.

# Note
- I'm using **Python Version:** *3.10.2* for this project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---
