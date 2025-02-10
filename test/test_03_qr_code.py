import cv2
from pyzbar.pyzbar import decode

def test_qr_code(image_path):
    image = cv2.imread(image_path)
    qr_codes = decode(image)

    if not qr_codes:
        print("❌ QR 코드 감지 실패")
    else:
        for qr in qr_codes:
            print(f"✅ QR 코드 감지됨: {qr.data.decode('utf-8')}")

test_qr_code("test.jpg")