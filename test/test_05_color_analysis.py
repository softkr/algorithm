import cv2
import numpy as np
import pandas as pd

# 🔹 색상 분석 함수
def estimate(colors):
    result = {'success': False}

    # 예제: RGB 데이터 (테스트용)
    test_ref_RGB = np.array([
        [212, 181, 153, 209, 214, 127, 96, 51, 215],
        [209, 178, 151, 204, 210, 126, 95, 51, 213],
        [197, 168, 144, 191, 199, 122, 96, 55, 201]
    ])

    # 색상 분석을 위한 예제 데이터
    categories = ['blood', 'protein', 'ketone', 'glucose', 'pH']
    colors_copy = colors.copy()

    # 🔹 KNN 모델 적용
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

# 🔹 테스트 실행 함수
def test_color_analysis():
    colors = {
        'blood': {'R': 150, 'G': 130, 'B': 120},
        'protein': {'R': 160, 'G': 140, 'B': 130},
        'ketone': {'R': 170, 'G': 150, 'B': 140},
        'glucose': {'R': 180, 'G': 160, 'B': 150},
        'pH': {'R': 190, 'G': 170, 'B': 160}
    }

    result = estimate(colors)

    if result["success"]:
        print("✅ 색상 분석 성공")
        print(f"결과: {result['levels']}")
    else:
        print("❌ 색상 분석 실패")

# 실행
test_color_analysis()