import os
import cv2
import mediapipe as mp
import numpy as np


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


if os.path.exists("Train") is False:
    os.makedirs("Train")
    os.makedirs(r"Train\Happy")
    os.makedirs(r"Train\Normal")
    os.makedirs(r"Train\Surprised")
    os.makedirs(r"Train\Disgust")
    os.makedirs(r"Train\Sad")


if os.path.exists("Val") is False:
    os.makedirs("Val")
    os.makedirs(r"Val\Happy")
    os.makedirs(r"Val\Normal")
    os.makedirs(r"Val\Surprised")
    os.makedirs(r"Val\Disgust")
    os.makedirs(r"Val\Sad")


if os.path.exists("Test") is False:
    os.makedirs("Test")
    os.makedirs(r"Test\Happy")
    os.makedirs(r"Test\Normal")
    os.makedirs(r"Test\Surprised")
    os.makedirs(r"Test\Disgust")
    os.makedirs(r"Test\Sad")


cap = cv2.VideoCapture(0)
face_mesh_mp = mp.solutions.face_mesh
face_mesh = face_mesh_mp.FaceMesh()
drawing_mp = mp.solutions.drawing_utils

class_name = "Disgust"
data_dir = "Train"
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if success is False:
        break

    lm_list = get_landmarks(img, draw=False)
    if lm_list:
        face_region = np.array(lm_list)
        min_x = np.min(face_region[:, 0])
        max_x = np.max(face_region[:, 0])
        min_y = np.min(face_region[:, 1])
        max_y = np.max(face_region[:, 1])

        img_cropped = img[min_y:max_y, min_x:max_x]
        cv2.imshow("Cropped", img_cropped)

    cv2.imshow("Res", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        print("Saved")
        base_path = rf"{data_dir}\{class_name}"
        cv2.imwrite(rf"{base_path}\{class_name}{len(os.listdir(base_path))}.jpg", img_cropped)

    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
