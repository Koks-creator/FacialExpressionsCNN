import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras


def get_landmarks(img_to_proc: np.array, draw=False) -> list:
    landmarks_list = []
    h, w, _ = img_to_proc.shape

    img_rgb = cv2.cvtColor(img_to_proc, cv2.COLOR_BGR2RGB)

    results_raw = face_mesh.process(img_rgb)
    results = results_raw.multi_face_landmarks

    if results:
        for landmark in results:
            if draw:
                drawing_mp.draw_landmarks(img_to_proc, landmark, face_mesh_mp.FACE_CONNECTIONS)

            for lm in landmark.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks_list.append((x, y))
    return landmarks_list


cap = cv2.VideoCapture(0)
face_mesh_mp = mp.solutions.face_mesh
face_mesh = face_mesh_mp.FaceMesh()
drawing_mp = mp.solutions.drawing_utils
model = keras.models.load_model('OwnDataModel.h5')

classes = ["Disgust", "Happy", "Normal", "Sad", "Surprised"]  # OwnDataModel.h5 classes
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if success is False:
        break

    lm_list = get_landmarks(img, draw=False)
    if lm_list:
        try:
            face_region = np.array(lm_list)
            min_x = np.min(face_region[:, 0])
            max_x = np.max(face_region[:, 0])
            min_y = np.min(face_region[:, 1])
            max_y = np.max(face_region[:, 1])

            img_cropped = img[min_y:max_y, min_x:max_x]
            img_cropped = cv2.resize(img_cropped, (224, 224))
            data = np.array([img_cropped])
            prediction = model.predict(data)
            label = np.argmax(prediction)

            print(classes[label])
            cv2.imshow("Cropped", img_cropped)
            cv2.putText(img, f"Class: {classes[label]}", (20, 50), 1, cv2.FONT_HERSHEY_COMPLEX, (255, 0, 255), 2)
        except Exception:
            pass

    cv2.imshow("Res", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
