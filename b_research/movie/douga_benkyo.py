import cv2

cap = cv2.VideoCapture('potechi8.mp4')

print('型::::::::::::::::::::::::::::::',type(cap))
print('読み取れたか::::::::::::::::::::',cap.isOpened())
print('動画の横幅::::::::::::::::::::::',cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('動画の縦幅::::::::::::::::::::::',cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('FPS(1秒あたりのフレーム数)::::::',cap.get(cv2.CAP_PROP_FPS))
print('総フレーム数::::::::::::::::::::',cap.get(cv2.CAP_PROP_FRAME_COUNT))

FPS = cap.get(cv2.CAP_PROP_FPS)
ALL = cap.get(cv2.CAP_PROP_FRAME_COUNT)
allsecond = ALL/FPS
print('再生時間::::::::::::::::::::::::',allsecond)

print('現在のフレーム数::::::::::::::::',cap.get(cv2.CAP_PROP_POS_FRAMES))
print('現在の秒数::::::::::::::::::::::',cap.get(cv2.CAP_PROP_POS_MSEC))

ret, frame = cap.read()

print('フレームが読み取れたかどうか::::',ret)
print('フレームの型::::::::::::::::::::',type(frame))
print('フレームの次元、型::::::::::::::',frame.shape)

print('現在のフレーム数::::::::::::::::',cap.get(cv2.CAP_PROP_POS_FRAMES))
print('現在の秒数::::::::::::::::::::::',cap.get(cv2.CAP_PROP_POS_MSEC))

cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
print('フレームを100に移動させた:::::::',cap.get(cv2.CAP_PROP_POS_FRAMES))
