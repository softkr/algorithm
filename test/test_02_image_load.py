import cv2

def test_image_load(image_path):
    image = cv2.imread(image_path)
    assert image is not None, f"❌ 이미지 로드 실패: {image_path}"
    print(f"✅ 이미지 로드 성공: {image_path}")

test_image_load("test.jpg")