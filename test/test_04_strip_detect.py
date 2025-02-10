import cv2
import numpy as np
import time
from scipy.optimize import least_squares

# 🔹 유틸리티 함수: 포인트 정렬
def sort_points(points):
    points = np.asarray(points)
    a = points[:, 0] + points[:, 1]
    b = points[:, 0] - points[:, 1]
    out = np.zeros((4, 2), dtype='f4')
    out[0] = points[a.argmin()]  # 좌측 상단
    out[1] = points[b.argmax()]  # 우측 상단
    out[2] = points[a.argmax()]  # 우측 하단
    out[3] = points[b.argmin()]  # 좌측 하단
    return out

# 🔹 스트립 감지 클래스
class Strip:
    def __init__(self, bbox_points, patch_points, patch_hsize, ref_points, strip_length, bbox_padding=0):
        self.bbox_points = sort_points(bbox_points)
        self.patch_points = patch_points
        self.patch_hsize = patch_hsize
        self.ref_points = ref_points
        self.strip_length = strip_length
        self.bbox_padding = bbox_padding

    def __call__(self, image, filename):
        start_time = time.time()

        if isinstance(image, str):
            image = cv2.imread(image)

        bbox = image.copy()
        gray = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
        result = {'bbox': bbox, 'gray': gray, 'success': False}
        blur = cv2.medianBlur(gray, 5)

        for ksize in [5, 10, 15]:
            if self.process(bbox, blur, ksize, result, filename):
                result['success'] = True
                break

        result['elapsed'] = time.time() - start_time
        return result

    def process(self, bbox, gray, ksize, result, filename):
        edge = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((ksize, ksize)))
        threshold, binary = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(approx)

            if len(approx) >= 4 and area > 1000:
                approx.shape = -1, 2
                pt0 = sort_points(np.array(approx, 'f4'))
                M = cv2.getPerspectiveTransform(pt0, self.bbox_points)
                roi = cv2.warpPerspective(bbox, M, (200, 200))

                result['bbox'] = roi
                return True

        return False

# 🔹 테스트 실행 함수
def test_strip_detection(image_path):
    bbox_points = np.array([[50, 50], [200, 50], [200, 200], [50, 200]], dtype=np.float32)
    patch_points = None
    patch_hsize = None
    ref_points = None
    strip_length = 150

    strip_instance = Strip(bbox_points, patch_points, patch_hsize, ref_points, strip_length)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return

    result = strip_instance(image, image_path)

    if result["success"]:
        print("✅ 스트립 탐색 성공")
    else:
        print("❌ 스트립 탐색 실패")

# 실행
test_strip_detection("test.jpg")