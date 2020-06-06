import cv2

# 読み込む動画の設定
# cf. https://docs.opencv.org/3.4.1/d8/dfe/classcv_1_1VideoCapture.html
cap = cv2.VideoCapture("./jui1.mp4")

while True:

    # フレームの読み込み
    ret, frame = cap.read()

    # 結果画像の表示
    cv2.imshow("frame", frame)
    k = cv2.waitKey(30) & 0xff
