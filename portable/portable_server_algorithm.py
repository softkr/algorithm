from ast import literal_eval
import sys
from turtle import color
import cv2
import json
import numpy as np
import pandas as pd


def estimate(colors:dict):
    # colors라는 dict는 단일항목의 R,G,B,C 값을 가진 dict이면 된다
    # 예) colors = { 'ketone' : { 'R':1234, 'G':1234, 'B':1234, 'C':1234 }}
    
    colors_copy = json.loads(colors)
    # 항목설정
    category = list(colors_copy.keys())[0]
    
    # 데이터 전처리
    Clear = int(colors_copy[category]['C'])
    R = int(colors_copy[category]['R'])
    G = int(colors_copy[category]['G'])
    B = int(colors_copy[category]['B'])
    Rg = (((R/Clear) * 256 / 255) ** 2.5) * 255
    Gg = (((G/Clear) * 256 / 255) ** 2.5) * 255
    Bg = (((B/Clear) * 256 / 255) ** 2.5) * 255
    # 트레이닝 데이터 불러오기
    # 레퍼런스 용지 모델 사용시
    train_data = pd.read_csv(sys.path[0] + '/portable_training_reference_220311.csv')
    # 시약 모델 사용시
    # train_data = pd.read_csv(sys.path[0] + '/portable_training_urine_220311.csv')
    rows = train_data['category'] == category
    data = train_data.loc[rows]
    
    data_rgb = data.loc[:,['R_gamma','G_gamma','B_gamma']]
    data_levels = data.loc[:,'expected']
    array_data_rgb = data_rgb.to_numpy().astype(np.float32)
    array_data_levels = data_levels.to_numpy().astype(np.int32)
    
    # 테스트 데이터 설정
    test_rgb = np.array([[Rg, Gg, Bg]])
    
    array_test_data_rgb = test_rgb.astype(np.float32)
    
    # SVM - 레벨계산
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.trainAuto(array_data_rgb, cv2.ml.ROW_SAMPLE, array_data_levels)

    # print('C:', svm.getC())
    # print('Gamma:', svm.getGamma())
    _,res = svm.predict(array_test_data_rgb)
    result = int(res[0][0])
    
    # 또는 dict로 리턴
    # result = { category: {'level': int(res[0][0])} }
    print(result, end='')
    return result
    
if __name__ == "__main__":
    try:
        estimate(sys.argv[1])
    except KeyboardInterrupt:
        pass