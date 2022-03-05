import cv2
import numpy as np
import pickle


face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")


with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # recognizing faces

        id_, conf = recognizer.predict(roi_gray)
        if conf > 45:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            names = labels[id_]
            color = (255, 255, 255)
            line_thickness = 2

            cv2.putText(
                frame, names, (x, y), font, 1, color, line_thickness, cv2.LINE_AA
            )

        color = (0, 255, 0)
        color_line_width = 2
        end_coordinate_x = x + w
        end_coordinate_y = y + h
        cv2.rectangle(
            frame,
            (x, y),
            (end_coordinate_x, end_coordinate_y),
            color=color,
            thickness=color_line_width,
        )

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
