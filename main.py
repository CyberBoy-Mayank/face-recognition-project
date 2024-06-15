import capture_images
import recognize_faces
import train_model


def main():
    while True:
        print("Choose an option:")
        print("1. Capture Images")
        print("2. Train Model")
        print("3. Recognize Faces and Mark Attendance")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            capture_images.capture_images()
        elif choice == '2':
            train_model.train_recognizer()
        elif choice == '3':
            recognize_faces.recognize_faces()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
