import cv2
import os


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def capture_images():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    face_id = input("Enter the unique ID of the person: ")
    name = input("Enter the name of the person: ")
    count = 0

    dataset_dir = 'dataset'
    create_directory(dataset_dir)

    csv_file = 'names.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('ID,Name\n')

    with open(csv_file, 'a') as f:
        f.write(f"{face_id},{name}\n")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()
