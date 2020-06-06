import numpy as np
import cv2

start = 0
sp_frame_num = 0
stop = False
move_start = 0

# 読み込む動画の設定
cap = cv2.VideoCapture("./douga2/pre_2.mp4")
model = './six_j.h5'

# Shi-Tomasiのコーナー検出パラメータ
feature_params = dict(
    maxCorners=255,             # 保持するコーナー数, int
    qualityLevel=0.0001,           # 最良値(最大固有値の割合?), double
    minDistance=7,              # この距離内のコーナーを棄却, double
    blockSize=7,                # 使用する近傍領域のサイズ, int
    useHarrisDetector=False,    # FalseならShi-Tomashi法
    # k=0.04,                     # Harris法の測度に使用
)

# Lucas-Kanade法のパラメータ
lk_params = dict(
    winSize=(15, 15),           # 検索ウィンドウのサイズ
    maxLevel=2,                 # 追加するピラミッド層数

    # 検索を終了する条件
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    ),

    # 推測値や固有値の使用
    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
)

# 何色でフローを描くか，色のリストを作る
color = np.random.randint(
    low=0,                  # 0から
    high=255,               # 255までの (輝度値なので0~255になります)
    size=(255, 3)           # 255(255個の特徴点を検出したいので)×3(RGBなので)の行列を作る
)

# 最初のフレームを読み込む
ret, first_frame = cap.read()

#動画の縦と横の長さとチャンネル数
h, w, c = first_frame.shape
print('')
print('動画の縦の長さ：',h)
print('動画の横の長さ：',w)
print('')

# グレースケール変換
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 読み込んだフレームの特徴点を探す
prev_points = cv2.goodFeaturesToTrack(
    image=first_gray,       # 入力画像
    mask=None,              # mask=0のコーナーを無視
    **feature_params
)

# 結果を描く画像のレイヤーを作る
flow_layer = np.zeros_like(first_frame)

# whileループで読み込むための準備
old_frame = first_frame
old_gray = first_gray

while True:

    x_max_list_1 = []
    x_min_list_1 = []
    y_max_list_1 = []
    y_min_list_1 = []

    start += 1
    if start == 50:
        print('start')
    move_num = 0

    # 2枚目以降のフレームの読み込み
    ret, frame = cap.read()

    # グレースケール変換
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # オプティカルフロー(正確には対応点)の検出
    # next_points: 検出した対応点, numpy.ndarray
    # status: 各点において，見つかれば1(True), 見つからなければ0(False), numpy.ndarray
    # err: 検出した点の誤差, numpy.ndarray
    next_points, status, err = cv2.calcOpticalFlowPyrLK(
        prevImg=old_gray,           # 前の画像(t-1)
        nextImg=frame_gray,         # 次の画像(t)
        prevPts=prev_points,        # 始点2次元ベクトル, 特徴点やそれに準ずる点
        nextPts=None,               # 結果の2次元ベクトル
        **lk_params
    )

    # 正しく特徴点と対応点が検出できた点のみに絞る
    # todo: 数フレームおきに特徴点を検出しなおさないと，対応点が無くなるのでエラーになります
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # フローを描く
    for rank, (prev_p, next_p) in enumerate(zip(good_old, good_new)):

        # x,y座標の取り出し
        # prev_x, prev_y: numpy.float32
        # next_x, next_y: numpy.float32
        prev_x, prev_y = prev_p.ravel()
        next_x, next_y = next_p.ravel()

        if abs(prev_y - next_y) >= 5:
            if abs(prev_x - next_x) < 10:

                move_num +=1
                if start >= 50:
                    stop = True
                    move_start += 1
                    if move_start == 1:
                        print('stop_OK')

                # フローの線を描く
                flow_layer = cv2.line(
                    img=flow_layer,                 # 描く画像
                    pt1=(prev_x, prev_y),           # 線を引く始点
                    pt2=(next_x, next_y),           # 線を引く終点
                    color=color[rank].tolist(),     # 描く色
                    thickness=2,                    # 線の太さ
                    # lineType=0,                   # 線の種類，無くても良い
                    # shift=0,                      # 無くても良い
                )
                # フローの特徴点を描く
                flow_layer = cv2.circle(
                    img=flow_layer,                 # 描く画像
                    center=(prev_x, prev_y),        # 円の中心
                    radius=5,                       # 円の半径
                    color=color[rank].tolist(),     # 描く色
                    thickness=1                     # 円の線の太さ
                )

                if prev_y > next_y:
                    y_max = prev_y
                    y_min = next_y
                else:
                    y_max = next_y
                    y_min = prev_y
                if prev_x > next_x:
                    x_max = prev_x
                    x_min = next_x
                else:
                    x_max = next_x
                    x_min = prev_x

                if x_max > w:
                    x_max = w
                if x_min < 0:
                    x_min = 0
                if y_max > h:
                    y_max = h
                if y_min < 0:
                    y_min = 0

                x_max_list_1.append(int(x_max))
                x_min_list_1.append(int(x_min))
                y_max_list_1.append(int(y_max))
                y_min_list_1.append(int(y_min))

    x_max_list_1.sort(reverse = True)
    x_min_list_1.sort()
    y_max_list_1.sort(reverse = True)
    y_min_list_1.sort()

    #はずれ値除去
    if len(x_max_list_1) >= 25:
        if x_max_list_1[0] - x_max_list_1[1] > 150:
            del x_max_list_1[0]
        if x_min_list_1[0] - x_min_list_1[1] < -150:
            del x_min_list_1[0]
        if y_max_list_1[0] - x_max_list_1[1] > 150:
            del x_max_list_1[0]
        if y_min_list_1[0] - x_min_list_1[1] < -150:
            del x_min_list_1[0]

        x_max_last = x_max_list_1[0]
        x_min_last = x_min_list_1[0]
        y_max_last = y_max_list_1[0]
        y_min_last = y_min_list_1[0]
        #print(len(x_max_list_1))

    if move_num == 0: #何も検出しないフレームの場合
        sp_frame_num += 1 #何も検出しないフレームの連続数
    else:
        sp_frame_num = 0 #動いたらリセット

    # 元の画像に重ねる
    result_img = cv2.add(frame, flow_layer)

    if stop == True and len(y_min_list_1) >= 1:
        result_img = cv2.line(
            img=result_img,                 # 描く画像
            pt1=(x_min_list_1[0], y_min_list_1[0]),           # 線を引く始点
            pt2=(x_max_list_1[0], y_min_list_1[0]),           # 線を引く終点
            color=color[rank].tolist(),     # 描く色
            thickness=2,                    # 線の太さ
            # lineType=0,                   # 線の種類，無くても良い
            # shift=0,                      # 無くても良い
        )
        result_img = cv2.line(
            img=result_img,                 # 描く画像
            pt1=(x_min_list_1[0], y_min_list_1[0]),           # 線を引く始点
            pt2=(x_min_list_1[0], y_max_list_1[0]),           # 線を引く終点
            color=color[rank].tolist(),     # 描く色
            thickness=2,                    # 線の太さ
            # lineType=0,                   # 線の種類，無くても良い
            # shift=0,                      # 無くても良い
        )
        result_img = cv2.line(
            img=result_img,                 # 描く画像
            pt1=(x_min_list_1[0], y_max_list_1[0]),           # 線を引く始点
            pt2=(x_max_list_1[0], y_max_list_1[0]),           # 線を引く終点
            color=color[rank].tolist(),     # 描く色
            thickness=2,                    # 線の太さ
            # lineType=0,                   # 線の種類，無くても良い
            # shift=0,                      # 無くても良い
        )
        result_img = cv2.line(
            img=result_img,                 # 描く画像
            pt1=(x_max_list_1[0], y_min_list_1[0]),           # 線を引く始点
            pt2=(x_max_list_1[0], y_max_list_1[0]),           # 線を引く終点
            color=color[rank].tolist(),     # 描く色
            thickness=2,                    # 線の太さ
            # lineType=0,                   # 線の種類，無くても良い
            # shift=0,                      # 無くても良い
        )

    # 結果画像の表示
    cv2.imshow("frame", result_img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if start >= 50 and sp_frame_num == 10 and stop == True:
        if int(y_min_last)-50 < 0:
            y_min_last = 0
        else:
            y_min_last = int(y_min_last)-50

        if int(y_max_last)+50 > h:
            y_max_last = h
        else:
            y_max_last = int(y_max_last)+50

        if int(x_min_last) < 0:
            x_min_last = 0
        else:
            x_min_last = int(x_min_last)

        if int(x_max_last)+50 > w:
            x_max_last = w
        else:
            x_max_last = int(x_max_last)+50

        triming = frame[y_min_last:y_max_last,x_min_last:x_max_last]
        cv2.imwrite('./triming.jpg',triming)
        cv2.imshow('triming', triming)
        cv2.waitKey(3000)
        break

    # 結果を描く画像のレイヤーを作る(リセットバージョン)
    flow_layer = np.zeros_like(first_frame)

    # 次のフレームを読み込む準備
    old_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()




"""
識別
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys
import os, glob


classes = ['jui', 'ocha', 'piza', 'pop', 'pre', 'toppo']
num_classes = len(classes)
IMAGE_SIZE = 120

# load model
model = load_model(model)
files = glob.glob('./triming.jpg')


for index, photos in enumerate(files):

    # convert data by specifying file from terminal
    #image = Image.open(sys.argv[1])
    image = Image.open(photos)
    image = image.convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    X = X.astype("float32")


    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    #print(photos[11:])
    print('')
    print(classes[predicted], percentage)
    print('')


"""
# convert data by specifying file from terminal
#image = Image.open(sys.argv[1])
image = Image.open("./test_img/ocha_3.jpg")
image = image.convert('RGB')
image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
data = np.asarray(image)
X = []
X.append(data)
X = np.array(X)
X = X.astype("float32")


# load model
model = load_model('./six.h5')

probs= model.predict(X)

print(probs)

#a = probs[0]
#print(a)

print('{:.10f}'.format(probs[0][0])) #小数点以下1位
print('{:.10f}'.format(probs[0][1])) #小数点以下2位
print('{:.10f}'.format(probs[0][2])) #小数点以下3位
print('{:.10f}'.format(probs[0][3])) #小数点以下3位
print('{:.10f}'.format(probs[0][4])) #小数点以下3位
print('{:.10f}'.format(probs[0][5])) #小数点以下3位

# estimated result of the first data (multiple scores will be returned)
#result = model.predict([X])[0]
#predicted = result.argmax()
#percentage = int(result[predicted] * 100)

#print(classes[predicted], percentage)
"""




import os
from gtts import gTTS

mytext0 = "ジュースです。"
mytext1 = "お茶です。"
mytext2 = "ピザポテトです。"
mytext3 = "ポップコーンです。"
mytext4 = "プリッツです。"
mytext5 = "トッポです。"


language = "ja"

if predicted == 0:
    mytext = mytext0
if predicted == 1:
    mytext = mytext1
if predicted == 2:
    mytext = mytext2
if predicted == 3:
    mytext = mytext3
if predicted == 4:
    mytext = mytext4
if predicted == 5:
    mytext = mytext5

myobj = gTTS(text=mytext, lang=language, slow=False)

myobj.save("syouhin.mp3")

os.system("mpg321 syouhin.mp3")
