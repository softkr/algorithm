
import numpy as np
import cv2
import time

from scipy.optimize import least_squares

from .utils import sort_points

class Strip:
    def __init__(self, bbox_points, patch_points, patch_hsize, ref_points, strip_length, bbox_padding=0):
        self.bbox_points = sort_points(bbox_points)
        self.patch_points = patch_points
        self.patch_hsize = patch_hsize
        self.ref_points = ref_points
        self.strip_length = strip_length
        self.bbox_padding = bbox_padding


    def __call__(self, image, filename, roi):
        start_time = time.time()

        # black_box.py 에서 받아온 bbox 이미지 읽기
        if isinstance(image, str):
            image = cv2.imread(image)
            
        bbox = image.copy()
        
        # bbox 흑백으로 바꾸기
        gray = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
        
        # result에 bbox, 흑백 bbox, success 여부 저장
        result = {
            'bbox': bbox,
            'gray': gray,
            'success': False
        }

        # 흑백 bbox를 블러처리 (노이즈 감소)
        blur = cv2.medianBlur(gray, 5)

        # 스트립 인식 / 색추출 될때까지 커널 사이즈 변경하며 시도
        for ksize in [5, 10, 15]:
            if self.process(bbox, blur, ksize,
                            self.patch_points,
                            self.patch_hsize,
                            self.ref_points,
                            self.strip_length,
                            self.bbox_padding,
                            result, filename, roi):

                result['success'] = True
                break
        
        result['elapsed'] = time.time() - start_time
        return result


    def process(self, bbox, gray, ksize, patch_points, patch_hsize, ref_points, strip_length, bbox_padding, result, filename, roi):
        
        # bbox를 out으로 설정 및 저장
        out = np.array(bbox)

        paddings = np.array([bbox_padding, bbox_padding*3])
        bbox_padding_points = np.array([[bbox_padding,bbox_padding],paddings,paddings*1.5,paddings*2,paddings*2.5]).astype(int)
        
        # 스트립의 기울기 판단 (회귀선 구함) - 인식이 될때까지 블랙박스를 조금씩 크롭해가며 실행. (빛반사 영향 최소화)
        for crop in bbox_padding_points:
            success, poly = self._fit(gray, ksize, crop)
            if success:
                bbox = bbox[crop[0]:-crop[0], crop[1]:-crop[1]]
                out = out[crop[0]:-crop[0], crop[1]:-crop[1]]
                gray = gray[crop[0]:-crop[0], crop[1]:-crop[1]]
                break
        else:
            return False
        
        # 스트립 기울기/위치에 넓이 부여
        X, Y = self._where(gray, poly, patch_hsize)
        out[Y, X] = (204, 255, 102)
        
        # 블랙박스 내의 그림자 보정
        bbox, gray, out = self._bbox_correction(bbox, gray, poly, patch_hsize, out)
        result['bbox'] = bbox.astype(np.float32)
        result['out'] = out.astype(np.float32)

        # 스트립의 엣지 찾기
        for length_errorrate in [0.0127, 0.0254, 0.0381]:
            for measure in [np.mean, np.median, np.max]:
                found, edges, line = self._edges(gray, X, Y, measure, poly, strip_length, result, out, length_errorrate)
                if found:
                    break
            else:
                continue
            break
        
        # 스트립 길이가 length validation 통과하지 못했을때 임의로 엣지 설정
        if not found:
            found, edges, line = self._edges_bent(gray, X, Y, result, out, poly, strip_length)
            if not found:
                return False

        # 스트립 방향 확인 및 항목 위치 설정
        successful, masks = self._masks_combined(edges, line, gray, X, Y, patch_points, patch_hsize, strip_length)
        if not successful:  
            result['error'] = 'NOSTRIP'
            return False
        result['masks'] = masks
        
        # 항목 색상 추출
        colors = self._colors(bbox, masks)
        result['colors'] = colors
        
        # 레퍼런스 (그레이스케일) 위치 계산 및 색상 추출
        h, w = roi.shape[:2]
        ref_masks = self._masks(h, w, ref_points, patch_hsize)
        ref_colors = self._colors(roi, ref_masks)
        result['colors'].update(ref_colors)
        
        # 레퍼런스 영역 사진에 표시
        roi = self._mark_ref(roi, ref_masks, ref_colors)
        
        # 사진 출력
        # original_name = filename.split('.')[0]
        # cv2.imwrite(original_name + '_reference.jpg', roi)
        # cv2.imwrite(original_name + '_mask.jpg', result['out'])

        return True
        

    def _fit(self, gray, ksize, crop):
        # bbox의 테두리를 크롭
        gray = gray[crop[0]:-crop[0], crop[1]:-crop[1]]
        # bbox(gray)사진 속 모양들의 edge만 살린 사진 생성
        edge = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((ksize, ksize)))
        # binary 사진으로 바꿀때 쓰이는 threshold 값을 구함
        threshold, binary = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # ix = edge.T(이미지 array)에서 threshold 보다 큰 값을 가진 좌표의 x index를 나열
        # iy = edge.T(이미지 array)에서 threshold 보다 큰 값을 가진 좌표의 y index를 나열. ix와 매칭됨
        ix, iy = np.array(np.where(edge.T > threshold))

        # edge사진에서 스트립이 발견되는 부분의 y index를 나열 (중복 X)
        ys = np.unique(iy)

        X = []
        Y = []
        # 스트립 길이를 따라 내려가면서 y좌표와 매칭되는 x좌표들의 평균값을 구함
        for y in ys:
            ind = np.where(iy == y)[0]
            X.append(np.mean(ix[ind]))
            Y.append(y)

        # 회귀선을 계산해서 스트립의 위치와 기울기를 구함
        err = lambda p, y, x: np.polyval(p, y) - x
        p0 = gray.shape[1] / 2, 0
        res = least_squares(err, p0, args=(Y, X), loss='soft_l1', f_scale=0.1)

        # 회귀선의 residual(에러율)이 너무 높으면 재시도 (보통 bbox가 덜 크롭됐거나 빛반사 때문에 회귀선이 잘못 잡혔을 시)
        if np.sum(abs(res.fun)) > 6000:
            return False, res.x
        else:
            return res.success, res.x


    def _where(self, gray, p, size): # p = 회귀식 계수, size = 색반응지 가로길이/2

        h, w = gray.shape

        m = np.arange(-size, size)
        n = -p[0] * m

        # 회귀선을 블랙박스 길이만큼 생성
        y = np.arange(0, h)
        x = np.polyval(p, y)

        # 회귀선에 색반응지 넓이만큼 넓이를 부여
        X = np.int32(np.tile(x, (m.size, 1)).T - m + 0.5)
        Y = np.int32(np.tile(y, (m.size, 1)).T - n + 0.5)

        # 블랙박스 범위를 넘어서는 index는 최소, 최대 범위로 변경
        X = np.clip(X, 0, w-1)
        Y = np.clip(Y, 0, h-1)

        return X, Y

    def _bbox_correction(self, bbox, gray, poly, patch_hsize, out):
        
        # 스트립 위치에서 좌우로 얼마나 옮겨가서 색상 인식할지 설정
        delta_x = np.hypot(59,(poly[0]*59))
        
        # 스트립 왼쪽의 블랙박스 값 구하기
        poly_left = np.copy(poly)
        poly_left[1] = poly_left[1] - delta_x
        left_X, left_Y = self._where(gray, poly_left, patch_hsize)
        left_line = np.mean(gray[left_Y, left_X], axis=1)
        
        # 스트립 오른쪽의 블랙박스 값 구하기
        poly_right = np.copy(poly)
        poly_right[1] = poly_right[1] + delta_x
        right_X, right_Y = self._where(gray, poly_right, patch_hsize)
        right_line = np.mean(gray[right_Y, right_X], axis=1)
        
        # # (확인용) 색상 표시
        # out[left_Y, left_X] = (255,0,255)
        # out[right_Y,right_X] = (0,255,0)
        
        # 왼쪽 오른쪽 평균
        shadow_line = np.mean(np.array([left_line, right_line]), axis=0, dtype=int)
        
        # 평균값 line의 0.75 percentile 값 구하기
        bbox_halfpoint = int(bbox.shape[0]/2)
        shadow_line_val = np.quantile(shadow_line[bbox_halfpoint:], 0.75, axis = 0)
        
        # 0.75 percentile 값과 평균값 line 값들의 차이를 구해서 bbox에 더해줌
        shadow_line_delta = - shadow_line + shadow_line_val
        shadow_line_delta = np.tile(shadow_line_delta, (bbox.shape[2],bbox.shape[1],1)).T
        bbox = bbox + shadow_line_delta
        bbox = np.clip(bbox, 0, 255)
        
        out = out + shadow_line_delta
        out = np.clip(out, 0, 255)
        
        return bbox, gray, out

    def _edges(self, gray, X, Y, measure, poly, length, result, out, length_errorrate):

        # 넓이를 부여했던 스트립 영역을 평균내서 다시 선으로 만듦
        line = measure(gray[Y, X], axis=1)
        x = np.arange(line.size)

        for threshold in self._threshold(line, x):
            # threshold와 line이 겹치는 부분을 구함
            rising = ((line[:-1] < threshold[:-1]) & (line[1:] > threshold[1:]))
            falling = ((line[:-1] > threshold[:-1]) & (line[1:] < threshold[1:]))

            if rising.any() and falling.any():
                # 겹치는 부분 중 처음과 마지막으로 겹치는 부분을 엣지로 설정
                edges = (np.flatnonzero(rising)[0] + 1,
                        np.flatnonzero(falling)[-1] + 1)

                if self._check_length(poly, edges, length, length_errorrate):
                    # 스트립 길이가 원본 스트립 길이와 비슷하면 엣지 확정 후 저장
                    out[Y[edges[0],:], X[edges[0],:]] = (0, 0, 255)
                    out[Y[edges[1],:], X[edges[1],:]] = (0, 0, 255)
                    result['out'] = out
                    
                    return True, edges, line

        return False, (0, 0), 0


    def _threshold(self, line, x):
        err = lambda p, x, y: np.polyval(p, x) - y
        p0 = (0, 0)

        for r0, r1 in [(0.2, 0.8), (0.6, 0.8), (0.2, 0.4)]:

            # 스트립 중간(0.2~0.8)의 값들에 대한 회귀선을 잡음
            x0 = int(x.size * r0)
            x1 = int(x.size * r1)
            res = least_squares(err, p0, args=(x[x0:x1], line[x0:x1]), loss='soft_l1', f_scale=0.1)
            
            if res.success:
                # 회귀선이 너무 기울어지는것을 방지
                res.x[0] = min(0.1,max(-0.1,res.x[0]))
                # 대략적인 스트립 밝기를 계산
                ind = np.argmin(line) # line에서 가장 작은 값을 가진 포인트의 index를 가져옴 (블랙박스 검정색 영역)
                delta = np.polyval(res.x, x[ind]) - line[ind] # 회귀선과 어두운 부분의 차이를 구함

                for ratio in [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1]:
                    # 회귀선을 여러가지 높이로 낮추어줌
                    p = res.x[0], res.x[1] - delta * ratio
                    yield np.polyval(p, x)
                    
        yield np.ones_like(line) * np.mean(line)
        yield np.ones_like(line) * np.median(line)

    def _check_length(self, p, edges, length, length_errorrate):
        # 측정된 스트립의 기울기를 감안해서 측정 스트립의 길이(L)를 계산
        dy = edges[1] - edges[0]
        dx = p[0] * dy
        L = np.hypot(dx, dy)
        
        # 측정된 스트립의 길이가 원래 스트립 길이 (80mm)에서 크게 벗어나지 않으면 넘어감
        return np.abs(L - length) / L < length_errorrate


    def _edges_bent(self, gray, X, Y, result, out, poly, length):
        
        line = np.mean(gray[Y, X], axis=1)
        
        # 스트립 미분선의 최고피크의 1/2 값을 넘는 포인트들 중 첫번째와 마지막 포인트를 엣지로 설정
        dy = np.diff(line)
        halfmax_dy = max(abs(dy)) * 0.5
        above_threshold = np.flatnonzero(abs(dy)>halfmax_dy)
        edges = (above_threshold[0] + 1), (above_threshold[-1] + 1)

        out[Y[edges[0],:], X[edges[0],:]] = (0, 0, 255)
        out[Y[edges[1],:], X[edges[1],:]] = (0, 0, 255)
        result['out'] = out
        
        # 최소한의 length validation
        dy = edges[1] - edges[0]
        dx = poly[0] * dy
        L = np.hypot(dx, dy)
        
        if np.abs(L - length) / max(1,L) < 0.1:
            return True, edges, line
        else: 
            return False, edges, line
    
    
    def _masks_combined(self, edges, line, gray, X, Y, points, size, length):
        
        # 스트립 위아래 판별
        L = edges[1] - edges[0]
        crop = 0.025 * L
        line_cropped = line[int(edges[0]+crop):int(edges[1]-crop)] # 스트립에서 인식한 양 끝에서 안으로 조금 크롭
        dy = abs(np.diff(line_cropped)) # 크롭된 스트립 line을 미분, 절댓값으로 변환
        half = int(line_cropped.size/2)
        dy_above_std = np.array(np.where((dy-np.average(dy)) > np.std(dy))) # 미분선에서 standard deviation을 넘는 포인트들을 구함
        
        masks = {}
        min_gray = min(line)
        
        # 스트립 정상적으로 붙였을 때 (색반응지가 밑으로)
        if (dy_above_std < half).sum() < (dy_above_std > half).sum():
            # 스트립 유무 판별 (스트립 손잡이 부분 값과 블랙박스의 제일 어두운 부분 값의 차이를 비교)
            strip_paper = np.average(line_cropped[:half])
            min_strip_diff = strip_paper - min_gray
            if min_strip_diff > 95:
                # 항목별 색반응지 위치 설정
                for name, center in points.items():
                    a = int(edges[1] - (center+size) / length * L)
                    b = int(edges[1] - (center-size) / length * L)
                    m = np.zeros_like(gray, dtype='?')
                    m[Y[a:b], X[a:b]] = True
                    masks[name] = m
                return True, masks
        # 스트립 반대로 붙였을 때 (색반응지가 위로)
        else:
            strip_paper = np.average(line_cropped[half:])
            min_strip_diff = strip_paper - min_gray
            if min_strip_diff > 95:
                for name, center in points.items():
                    a = int(edges[0] + (center-size) / length * L)
                    b = int(edges[0] + (center+size) / length * L)
                    m = np.zeros_like(gray, dtype='?')
                    m[Y[a:b], X[a:b]] = True
                    masks[name] = m
                return True, masks
            
        return False, masks
    
    def _colors(self, image, masks):
        
        colors = {}
        for name, mask in masks.items():
            # 항목 위치 인식하고 그 안의 색상 계산
            box = image[mask]
            mean = np.mean(box, axis=0)
            var = np.mean(np.var(box, axis=0))
            # 결과 저장
            colors[name] = {
                'B': mean[0],
                'G': mean[1],
                'R': mean[2],
                'std': np.sqrt(var)
            }
            
        return colors

    def _masks(self, h, w, points, hsize):
        
        masks = {}
        # 레퍼런스마다 위치를 인식해둠
        for name, center in points.items():
            a = (center-hsize).astype(int)
            b = (center+hsize).astype(int)
            m = np.zeros((h, w), dtype='?')
            m[a[1]:b[1], a[0]:b[0]] = True
            masks[name] = m
            
        return masks

    def _mark_ref(self, roi, masks, colors):
        
        # roi 사진에 인식된 레퍼런스 위치 표시하고 R,G,B 평균값 작성
        for name, mask in masks.items():
            m = mask.nonzero()
            avg_color = round((colors[name]['R']+colors[name]['G']+colors[name]['B'])/3)
            cv2.rectangle(roi, (m[1][0], m[0][0]), (m[1][-1], m[0][-1]), (255, 255, 0), 2)
            cv2.putText(roi, f'{name}){avg_color}', (m[1][0]-50, m[0][0]-10), cv2.FONT_HERSHEY_SIMPLEX , 1, ( 0, 0, 255), 2)
            
        return roi