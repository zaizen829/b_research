import cv2
import numpy as np


cap = cv2.VideoCapture("./douga/pop_j.mp4")
i = 1350

while True:

    i+=1
    # フレームの読み込み
    ret, frame = cap.read()

    # 結果画像の表示
    cv2.imshow("frame", frame)
    cv2.imwrite("./dataimg/pop/pop_" + str(i) + ".jpg",frame)
    k = cv2.waitKey(1)
    if i == 1500:
        break
