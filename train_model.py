import cv2
import os
import numpy as np


def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    data_path = 'dataset'
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        face_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y + h, x:x + w])
            ids.append(face_id)

    recognizer.train(face_samples, np.array(ids))
    recognizer.save('trainer.yml')
