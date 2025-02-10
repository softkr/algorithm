import cv2
import numpy as np
import time
from math import dist
import os

from pyzbar import pyzbar

from .utils import sort_points


class BlackBox:
    def __init__(self, bbox_points, roi_size, qr_centers, qr_size, min_area_ratio=0.01):
        self.bbox_points = bbox_points
        self.roi_size = roi_size
        self.qr_centers = qr_centers
        self.qr_size = qr_size
        self.min_area_ratio = min_area_ratio

    def __call__(self, image):
        if isinstance(image, str):
            filename = str(image)
            image = cv2.imread(image)

        return self.crop(image, filename,
                         self.bbox_points,
                         self.roi_size,
                         self.qr_centers,
                         self.qr_size,
                         self.min_area_ratio)

    def crop(self, bgr, filename, bbox_points, roi_size, qr_centers, qr_size, min_area_ratio=0.01):
        start_time = time.time()

        # bbox 좌표를 정렬해둠
        points = sort_points(bbox_points)

        return {}

        # bgr(이미지)에서 가로 세로 길이를 구함
        h, w = bgr.shape[:2]
        # 가로가 세로보다 길면 90도 회전
        if h < w:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 사진을 회색으로 변경
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # 회색 사진을 블러처리 (노이즈 줄임)
        blur = cv2.medianBlur(gray, 5)

        # 관심영역의 최소넓이를 계산해놓음 (사진 면적의 1% 이상)
        min_area = gray.size * min_area_ratio

        # result에 success 여부와 회색 사진을 넣어둠
        result = {
            'success': False,
            'gray': gray,
        }

        # 관심영역을 찾을때까지 커널사이즈를 바꾸어 가며 시도
        for ksize in [5, 10, 15, 20]:
            result['ksize'] = ksize
            # 사진 속 모양들의 edge만 살린 사진 생성
            edge = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((ksize, ksize)))
            # edge 사진을 binary로 변경
            threshold, binary = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # bbox, roi, qr 추출
            success = self._contour(result, bgr, filename, binary, points, roi_size, qr_centers, qr_size, min_area)
            result['success'] = success

            if success:
                break

        result['elapsed'] = time.time() - start_time

        return result

    def _contour(self, result, bgr, filename, binary, points, roi_size, qr_centers, qr_size, min_area):

        # binary 사진에서 contour들 찾기
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        max_matching = 0

        # 인식된 contour 마다 옆에 QR이 있는지 체크
        for contour in contours:
            # contour의 대략적 모양(approx)을 구함
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 대략적 모양의 넓이 구함
            area = cv2.contourArea(approx)

            # 대략적 모양의 변의 개수가 4개 이상이고 (사각형 이상) min area(사진의 1% 면적)를 넘는 contour만 선택
            if len(approx) >= 4 and area > min_area:

                # cv2.drawContours(out, [approx], 0, (0, 255, 255), 6) # 사진에 모양 그려보기
                approx.shape = -1, 2

                # 모양의 꼭짓점을 시계방향으로 정렬
                pt0 = sort_points(np.array(approx, 'f4'))

                # 모양의 가로 변이 세로 변보다 길면 시계방향으로 90도 회전
                if dist(pt0[0], pt0[1]) > dist(pt0[1], pt0[2]):
                    pt0_copy = pt0.copy()
                    pt0[0] = pt0_copy[1]
                    pt0[1] = pt0_copy[2]
                    pt0[2] = pt0_copy[3]
                    pt0[3] = pt0_copy[0]

                # 사진을 정상일때(pt0)와 뒤집혔을때(pt1) 좌표로 구분
                pt1 = np.zeros((4, 2), dtype='f4')
                pt1[0] = pt0[2]
                pt1[1] = pt0[3]
                pt1[2] = pt0[0]
                pt1[3] = pt0[1]

                # 사진이 뒤집혀 있을때 뒤집어서 정상으로 만든 후에 크롭 진행 / 사진이 뒤집혀 있지 않을 때에는 정상적으로 크롭 진행
                for i, pts in enumerate([pt0, pt1]):
                    # 도형을 urine config에서 설정해둔 bbox_points 좌표에 맞추는 transfomation matrix 생성
                    M = cv2.getPerspectiveTransform(pts, points)
                    # transformation matrix를 적용해서 펼쳐주고 사이즈를 설정해줌
                    roi = cv2.warpPerspective(bgr, M, roi_size)
                    # roi 좌표 찍어줌 (블랙박스 기준 좌우상하)
                    x0, y0 = np.int32(points[0])
                    x1, y1 = np.int32(points[2])
                    # roi에 qr이 있는지 확인
                    qr = self._test_qr(roi, qr_centers, qr_size)

                    if len(qr) > 0:
                        # roi template과 인식된 roi의 유사도 판별
                        template = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/roi_template.jpg')

                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                        res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCORR_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                        # 유사도가 높아질때마다 roi 업데이트
                        if max_val > max_matching:
                            max_matching = max_val
                            result['roi'] = roi
                            result['bbox_points'] = pts
                            result['qr'] = qr
                            bbox = roi[y0:y1, x0:x1]
                            result['bbox'] = bbox
                            result['orientation'] = i  # pt0 일때(=정상적인 사진) 와 pt1(=뒤집힌 사진) 일때 구분

                # 최종 roi의 유사도가 0.9 이상이면 넘어감
                if max_matching >= 0.9:
                    # 원본사진 크롭함
                    # 인식된 블랙박스 좌표 불러옴 (평면보정 전 좌표)
                    pnt0, pnt1, pnt2, pnt3 = result['bbox_points']

                    if result.get('orientation') == 1:  # 뒤집힌 사진의 경우 평면 보정 전 좌표를 바꿈
                        pnt0, pnt1, pnt2, pnt3 = pnt2, pnt3, pnt0, pnt1

                    xl = int(min(pnt0[0], pnt3[0]))
                    xr = int(max(pnt1[0], pnt2[0]))
                    yt = int(min(pnt0[1], pnt1[1]))
                    yb = int(max(pnt3[1], pnt2[1]))

                    # 4개 모서리에서 넓이 300 픽셀, 길이 150 필셀씩 늘려서 저장
                    xl = int(max(0, xl - 300))
                    xr = int(min(bgr.shape[1], xr + 350))
                    yt = int(max(0, yt - 150))
                    yb = int(min(bgr.shape[0], yb + 150))

                    cropped_image_path = str(filename.split('.')[0]) + '_crop.jpg'

                    cropped_img = bgr[yt:yb, xl:xr]
                    if result.get('orientation') == 1:
                        turned_cropped_image = cv2.rotate(cropped_img, cv2.ROTATE_180)
                        cv2.imwrite(cropped_image_path, turned_cropped_image)
                    else:
                        cv2.imwrite(cropped_image_path, cropped_img)

                    return len(result['qr']) > 0

        return False

    def _test_qr(self, bgr, centers, size):
        # roi를 흑백으로 변경
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # roi 가로 세로
        h, w = gray.shape

        hsize = size[0] / 2, size[1] / 2

        for ksize in [1, 3, 5, 7, 9]:
            # qr 의 4모서리 좌표 계산
            xc, yc = centers
            x0 = int(max(xc - hsize[0], 0))
            y0 = int(max(yc - hsize[1], 0))
            x1 = int(min(xc + hsize[0], w))
            y1 = int(min(yc + hsize[1], h))

            area = gray[y0:y1, x0:x1]
            # median blur로 이미지 전처리
            if ksize > 1:
                area = cv2.medianBlur(area, ksize)

            decoded = pyzbar.decode(area, symbols=[pyzbar.ZBarSymbol.QRCODE])
            # 인식이 되면 끝내기
            if len(decoded) > 0:
                return decoded[0].data.decode()

            # 인식이 안되면 binary로 전처리 하고 인식해보기
            _, binary = cv2.threshold(area, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            decoded = pyzbar.decode(binary, symbols=[pyzbar.ZBarSymbol.QRCODE])
            if len(decoded) > 0:
                return decoded[0].data.decode()

        return ''
