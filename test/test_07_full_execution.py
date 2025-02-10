import os
import cv2
import numpy as np
import pandas as pd

# 🔹 색상 분석 및 KNN 모델 적용 (독립 실행 가능하도록 수정)
def estimate(colors):
    result = {'success': False}

    # 테스트용 RGB 데이터
    test_ref_RGB = np.array([
        [212, 181, 153, 209, 214, 127, 96, 51, 215],
        [209, 178, 151, 204, 210, 126, 95, 51, 213],
        [197, 168, 144, 191, 199, 122, 96, 55, 201]
    ])

    categories = ['blood', 'protein', 'ketone', 'glucose', 'pH']
    colors_copy = colors.copy()

    # 🔹 KNN 모델 학습 및 예측
    train_data = pd.DataFrame({
        'R': [100, 150, 200, 250],
        'G': [100, 150, 200, 250],
        'B': [100, 150, 200, 250],
        'expected': [0, 1, 2, 3]
    })

    knn_dist = {}
    levels = {}

    for category in categories:
        data_rgb = train_data[['R', 'G', 'B']].to_numpy().astype(np.float32)
        data_levels = train_data['expected'].to_numpy().astype(np.int32)
        test_data = np.array([[colors_copy.get(category, {'R': 150, 'G': 150, 'B': 150})[ch] for ch in ['R', 'G', 'B']]])
        knn = cv2.ml.KNearest_create()
        knn.train(data_rgb, cv2.ml.ROW_SAMPLE, data_levels)
        _, results, _, dist = knn.findNearest(test_data.astype(np.float32), 5)
        knn_dist[category] = np.average(dist)
        levels[category] = int(results[0, 0])

    result['success'] = True
    result['levels'] = levels
    return result

# 🔹 전체 실행 테스트
def test_full_execution():
    image_path = "test.jpg"
    output_csv = "output/test.csv"
    output_img = "test_output.jpg"

    # 이미지 로드 테스트
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return

    # 가상의 색상 데이터
    colors = {
        'blood': {'R': 150, 'G': 130, 'B': 120},
        'protein': {'R': 160, 'G': 140, 'B': 130},
        'ketone': {'R': 170, 'G': 150, 'B': 140},
        'glucose': {'R': 180, 'G': 160, 'B': 150},
        'pH': {'R': 190, 'G': 170, 'B': 160}
    }

    # 색상 분석 실행
    result = estimate(colors)

    if result["success"]:
        print("✅ 전체 실행 성공")
        print(f"결과: {result['levels']}")
    else:
        print("❌ 전체 실행 실패")

    # 결과 저장 (CSV)
    os.makedirs("output", exist_ok=True)
    pd.DataFrame([result['levels']]).to_csv(output_csv, index=False)
    if os.path.exists(output_csv):
        print(f"✅ 결과 CSV 생성됨: {output_csv}")
    else:
        print("❌ 결과 CSV 생성 실패")

    # 결과 이미지 저장 (임시)
    cv2.imwrite(output_img, image)
    if os.path.exists(output_img):
        print(f"✅ 결과 이미지 생성됨: {output_img}")
    else:
        print("❌ 결과 이미지 생성 실패")

# 실행
test_full_execution()