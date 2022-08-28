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


def process_image(img_to_process: np.array, landmark_list: list):
    face_conts_region = np.array([landmark_list[10], landmark_list[338], landmark_list[297], landmark_list[332], landmark_list[284], landmark_list[251],
                                  landmark_list[389], landmark_list[366], landmark_list[401], landmark_list[435], landmark_list[367], landmark_list[364],
                                  landmark_list[379], landmark_list[400], landmark_list[152], landmark_list[176], landmark_list[150], landmark_list[135],
                                  landmark_list[138], landmark_list[215], landmark_list[177], landmark_list[137], landmark_list[162], landmark_list[21],
                                  landmark_list[54], landmark_list[103], landmark_list[67], landmark_list[109]])

    height, width, _ = img_to_process.shape
    blank_img = np.zeros((height, width), np.uint8)

    cv2.polylines(blank_img, [face_conts_region], True, 255, 2)
    cv2.fillPoly(blank_img, [face_conts_region], 255)
    face_raw = cv2.bitwise_and(img, img, mask=blank_img)

    min_x = np.min(face_conts_region[:, 0])
    max_x = np.max(face_conts_region[:, 0])
    min_y = np.min(face_conts_region[:, 1])
    max_y = np.max(face_conts_region[:, 1])

    face = face_raw[min_y:max_y, min_x:max_x]

    return face


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

class_name = "Sad"
data_dir = "train2"
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if success is False:
        break

    lm_list = get_landmarks(img, draw=False)
    if lm_list:

        face = process_image(img, lm_list)
        cv2.imshow("Cropped", face)

    cv2.imshow("Res", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        print("Saved")
        base_path = rf"{data_dir}\{class_name}"
        cv2.imwrite(rf"{base_path}\{class_name}{len(os.listdir(base_path))}.jpg", face)

    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
