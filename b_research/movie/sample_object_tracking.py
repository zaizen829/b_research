"""
#オプティカルフロー検出のサンプルプログラム
#大元: opencv3.2.0-samples-python
#ページ番号は参考となる詳解OpenCV3のページ番号
"""
import numpy as np
import cv2

# 読み込む動画の設定
# https://docs.opencv.org/3.4.1/d8/dfe/classcv_1_1VideoCapture.html
cap = cv2.VideoCapture("jui1.mp4")

# Shi-Tomasiのコーナー検出パラメータ
# P.511,477
feature_params = dict(
    maxCorners=255,             # 保持するコーナー数, int
    qualityLevel=0.3,           # 最良値(最大固有値の割合?), double
    minDistance=7,              # この距離内のコーナーを棄却, double
    blockSize=7,                # 使用する近傍領域のサイズ, int
    useHarrisDetector=False,    # FalseならShi-Tomashi法
    # k=0.04,                     # Harris法の測度に使用
)

# Lucas-Kanade法のパラメータ
# P.489
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
# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.randint.html
color = np.random.randint(
    low=0,                  # 0から
    high=255,               # 255までの (輝度値なので0~255になります)
    size=(255, 3)           # 255(255個の特徴点を検出したいので)×3(RGBなので)の行列を作る
)

# 最初のフレームを読み込む
ret, first_frame = cap.read()

# グレースケール変換
# P.111
# https://docs.opencv.org/3.4.1/d7/d1b/group__imgproc__misc.html#ga397ae87e1288a81d2363b61574eb8cab
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 読み込んだフレームの特徴点を探す
# P.477
# https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
prev_points = cv2.goodFeaturesToTrack(
    image=first_gray,       # 入力画像
    mask=None,              # mask=0のコーナーを無視
    **feature_params
)

# 結果を描く画像のレイヤーを作る
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros_like.html
flow_layer = np.zeros_like(first_frame)

# whileループで読み込むための準備
old_frame = first_frame
old_gray = first_gray

while True:

    # 2枚目以降のフレームの読み込み
    ret, frame = cap.read()

    # グレースケール変換
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # オプティカルフロー(正確には対応点)の検出
    # P.489
    # https://docs.opencv.org/3.4.1/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
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

        # フローの線を描く
        # https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
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
        # https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
        flow_layer = cv2.circle(
            img=flow_layer,                 # 描く画像
            center=(prev_x, prev_y),        # 円の中心
            radius=5,                       # 円の半径
            color=color[rank].tolist(),     # 描く色
            thickness=1                     # 円の線の太さ
        )

    # 元の画像に重ねる
    # https://docs.opencv.org/3.4.1/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6
    result_img = cv2.add(frame, flow_layer)

    # 結果画像の表示
    cv2.imshow("frame", result_img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # 次のフレームを読み込む準備
    old_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
