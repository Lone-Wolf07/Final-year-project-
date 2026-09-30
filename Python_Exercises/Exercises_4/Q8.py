import os
import face_recognition
import cv2

output_dir = os.path.dirname(os.path.abspath(__file__))

known_path = input("Enter known face image path: ")
unknown_path = input("Enter unknown image path: ")

known_image = face_recognition.load_image_file(known_path)
known_encoding = face_recognition.face_encodings(known_image)[0]

image = face_recognition.load_image_file(unknown_path)
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

frame = cv2.imread(unknown_path)

for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces([known_encoding], encoding)
    name = "Match" if matches[0] else "Unknown"
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

output_path = os.path.join(output_dir, "recognized.jpg")
cv2.imwrite(output_path, frame)
print("Face recognition complete, saved to", output_path)
