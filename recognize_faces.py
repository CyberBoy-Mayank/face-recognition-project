import cv2
import pandas as pd
from FaceMeshModule import FaceMeshDetector


def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    df = pd.read_csv('names.csv')
    id_name_map = {row['ID']: row['Name'] for _, row in df.iterrows()}

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
                name = id_name_map.get(id, "Unknown")
            else:
                name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"
            if name == "Unknown":
                cv2.putText(frame, f"{name}", (x + 5, y - 5), font, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"{name}: {confidence_text}", (x + 5, y - 5), font, 1, (0, 255, 0), 2)
        detector = FaceMeshDetector(maxFaces=2)
        detector.findFaceMesh(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_faces()
