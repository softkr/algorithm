import numpy as np
import urine_strip as us

# -----------------------------------------------------------------------------
# for black box analysis
# 기본 pixel은 300 dpi로 맞추어짐
# -----------------------------------------------------------------------------
# roi와 블랙박스 좌표

bbox_points = np.array([
    [us.mm2px(12), us.mm2px(5)],
    [us.mm2px(12 + 20), us.mm2px(5)], 
    [us.mm2px(12 + 20), us.mm2px(5 + 90)],
    [us.mm2px(12), us.mm2px(5 + 90)]
], dtype='f4')

roi_size = np.array([us.mm2px(47), us.mm2px(100)])

# -----------------------------------------------------------------------------
# QR 위치 및 크기 

qr_centers = np.array(
    [us.mm2px(39.5), us.mm2px(10.5)] # QR: 정상 위치일 때
)
qr_size = np.array([us.mm2px(14), us.mm2px(14)])

# -----------------------------------------------------------------------------
# for strip analysis 항목 순서

patch_points = {
    'ketone': us.mm2px(5),
    'glucose': us.mm2px(12.5),
    'protein': us.mm2px(20),
    'pH': us.mm2px(27.5),
    'blood': us.mm2px(35),
    'white': us.mm2px(42.5)
}

patch_hsize = us.mm2px(1.5)    # half size
strip_length = us.mm2px(80)
bbox_padding = 6

# -----------------------------------------------------------------------------
# 그레이스케일 레퍼런스 위치

ref_points = { # 좌측상단부터 시계반대방향
    'ref1': [us.mm2px(7),us.mm2px(55)],
    'ref2': [us.mm2px(7),us.mm2px(55+7.5)],
    'ref3': [us.mm2px(7),us.mm2px(55+(7.5*2))],
    'ref4': [us.mm2px(7),us.mm2px(55+(7.5*3))],
    'ref5': [us.mm2px(7),us.mm2px(55+(7.5*4))],
    'ref6': [us.mm2px(37),us.mm2px(55+(7.5*4))],
    'ref7': [us.mm2px(37),us.mm2px(55+(7.5*3))],
    'ref8': [us.mm2px(37),us.mm2px(55+(7.5*2))],
    'ref9': [us.mm2px(37),us.mm2px(55+7.5)],
    'ref10': [us.mm2px(37),us.mm2px(55)]
}

# -----------------------------------------------------------------------------

# 220510 양산본 그레이 스케일 모델 레퍼런스
# 첫째줄 = R, 둘째줄 = G, 셋째줄 = B
# 왼쪽에서 차례대로 ref1, ref2, ref3, ref4, ref5, ref6, ref7, ref8, ref9, ref10
model_ref_RGB = np.array([[212.2,	212.3,	181.1,	153.0,	209.0,	214.6,	127.0,	96.1,	51.6,	215.4],
                          [209.9,	209.5,	178.2,	151.1,	204.3,	210.9,	126.3,	95.6,	51.0,	213.3],
                          [197.7,	196.8,	168.9,	144.7,	191.4,	199.2,	122.5,	96.0,	55.0,	201.3]
                          ])