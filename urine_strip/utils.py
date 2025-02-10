import os
import glob as _glob
import cv2
import numpy as np


def rgb(image):
    # OpenCV represents images in BGR order; however, Matplotlib
    # expects the image in RGB order, so simply convert from BGR
    # to RGB and return
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sharpness(image):
    # Lapacian을 사용해서 sharpness 계산
    # 참고: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    return cv2.Laplacian(image, cv2.CV_32F).var()


def mkdir_p(path):
    # 디렉토리가 없는 경우 새로 만듬
    try:
        os.makedirs(path, exist_ok=True)  # python >= 3.2
    except OSError as exc:  # Python > 2.5
        raise  # 기존 디렉토리에 접근이 불가능 한 경우


def glob(filename):
    files = _glob.glob(filename)
    return [fn.replace('\\', '/') for fn in files]


def convert_path(infile, outdir, prefix=''):
    # infile의 directory를 outdir 로 변경
    _, fn = os.path.split(infile)
    return os.path.join(outdir, prefix + fn)


def sort_points(points):
    # 4개의 x, y 좌표인 points를 왼쪽상단 부터 시계방향으로 정렬
    points = np.asarray(points)

    a = points[:, 0] + points[:, 1]
    b = points[:, 0] - points[:, 1]

    out = np.zeros((4, 2), dtype='f4')

    out[0] = points[a.argmin()]  # x+y가 최솟값, 좌측상단
    out[1] = points[b.argmax()]  # x-y가 최댓값, 우측상단
    out[2] = points[a.argmax()]  # x+y가 최댓값, 우측하단
    out[3] = points[b.argmin()]  # x-y가 최솟값, 좌측하단

    return out


def mm2px(mm, dpi=300):
    # mm 를 pixel 크기로 변환
    return np.int32(round(dpi / 25.4 * mm))  # round() 사용
