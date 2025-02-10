import os
import cv2
import pandas as pd
import numpy as np


def estimate(model_ref_RGB, colors: dict, bbox, masks, filename):
    
    result = {
        'success': False
    }

    colors_copy = colors.copy()

    # 레퍼런스 R,G,B를 numpy array로 생성
    references = ['ref1', 'ref2', 'ref3','ref4','ref5','ref6','ref7','ref8','ref9','ref10']
    test_ref_RGB = np.empty([3,len(references)]) # [0, :] = red, [1, :] = green, [2, :] = blue
    rgb_color = ['R','G','B']
    for ref in references:
        for c in range(0,3):
            test_ref_RGB[c,references.index(ref)] = colors_copy[ref][rgb_color[c]]
    
    # shadow compensation - 4개 코너 값으로 상대적인 어둡기를 구하고 그레이 스케일을 그에 맞게 보정해줌
    max_ref_R = max(test_ref_RGB[0,[0,4,5,9]])
    Rm0 = max_ref_R - test_ref_RGB[0,0]
    Rm4 = max_ref_R - test_ref_RGB[0,4]
    Rm5 = max_ref_R - test_ref_RGB[0,5]
    Rm9 = max_ref_R - test_ref_RGB[0,9]
    for c in range(0,3):
        test_ref_RGB[c,1] += Rm0
        test_ref_RGB[c,2] += ((Rm0+Rm4)/2)
        test_ref_RGB[c,3] += Rm4
        test_ref_RGB[c,6] += Rm5
        test_ref_RGB[c,7] += ((Rm5+Rm9)/2)
        test_ref_RGB[c,8] += Rm9
    
    # 필요한 레퍼런스 선택
    grayscale_idx = np.array([8,7,6,3,2,1])
    corner_idx = np.array([0,4,5,9])
    
    x = model_ref_RGB[0,grayscale_idx] # model_ref_RGB 의 R값만 사용
    y_rgb = test_ref_RGB[:,grayscale_idx]
    corners = test_ref_RGB[:,corner_idx]
    
    # 역전되는 레퍼런스가 있으면 그 전과 후 레퍼런스 값의 평균을 구한 값으로 설정
    y_rgb_average = np.average(y_rgb, axis=0)
    if y_rgb_average[0] > y_rgb_average[1]: # 첫번째 레퍼런스는 두번째와 세번째 값의 차이를 두번째에서 뺸 값으로 설정
        for c in range(0,3):
            y_rgb[c,0] = (2 * y_rgb[c,1]) - y_rgb[c,2]
    for idx in range(1,5):
        if y_rgb_average[idx] > y_rgb_average[idx+1]:
            for c in range(0,3):
                y_rgb[c,idx] = (y_rgb[c,idx-1] + y_rgb[c,idx+1]) / 2
    
    # 그레이스케일 보정 (3차식 회귀)
    residuals = {}
    categories = ['blood','protein','ketone','glucose','pH']
    for c in range(0,3):
        y = y_rgb[c,:]
        pol, resid, _, _, _  = np.polyfit(y,x,3,full=True)
        residuals[str(c)] = resid
        for cat in categories:
            colors_copy[cat][rgb_color[c]] = np.polyval(pol,colors_copy[cat][rgb_color[c]])
    
    # 에러처리
    max_residual_key = max(residuals, key=residuals.get)
    max_residual = residuals[max_residual_key]
    if max_residual > 1000: # 회귀선이 잘 안 잡혔을 시
        result['error'] = 'REF'
        return {}, result
    corners_avg = np.average(corners, axis=0)
    if max(corners_avg) < 150: # 4개 코너 중 max가 150 이하면 에러 (너무 어두운 환경)
        result['error'] = 'SHADOW'
        return {}, result
    if max(corners_avg) - min(corners_avg) > 95: # 4개 코너간의 값 차이가 너무 많이 나면 에러 (그림자가 졌다고 판단)
        result['error'] = 'SHADOW'
        return {}, result


    # KNN 계산 - training data csv 불러오기
    train_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/220620_SU_training_data.csv')
    
    levels = {}
    knn_dist = {}
    
    for category in categories:
        
        # training data 설정
        rows = train_data['category'] == category
        data = train_data.loc[rows]
        data_rgb = data.loc[:,['R','G','B']]
        data_levels = data.loc[:,'expected']
        array_data_rgb = data_rgb.to_numpy().astype(np.float32)
        array_data_levels = data_levels.to_numpy().astype(np.int32)
        
        # testing data 설정
        test_data = colors_copy[category]
        test_data_list = np.array([[test_data['R'],test_data['G'],test_data['B']]])
        array_test_data_rgb = test_data_list.astype(np.float32)

        # train KNN
        knn = cv2.ml.KNearest_create()
        knn.train(array_data_rgb, cv2.ml.ROW_SAMPLE, array_data_levels)

        # test KNN (K 이웃 수 = 5)
        _, results, _, dist = knn.findNearest(array_test_data_rgb, 5)
        
        # 이웃들과의 평균 색거리 저장
        knn_dist[category] = np.average(dist)
        
        # 계산된 레벨 결과 저장
        levels[category] = int(results[0,0])
        

    # 색거리가 너무 멀게 나오면 에러처리 (색이 과보정/역보정 되었다고 판단)
    knn_dist_list = list(knn_dist.values())
    knn_dist_array = np.array(knn_dist_list)
    overdist = (knn_dist_array > 5000).sum()
    if overdist >= 3: # 색거리가 5,000이 넘는 항목이 3개 이상이면 에러
        result['error'] = 'REF'
        return {}, result
    elif np.any(knn_dist_array > 50000): # 색거리가 50,000이 넘는 항목이 하나라도 있으면 에러
        result['error'] = 'REF'
        return {}, result
    
    # 색보정된 마스크 사진 출력
    for name, mask in masks.items():
        # 보정된 색을 0-255 사이 값으로 조정
        r = max(0,min(255,colors_copy[name]['R']))
        g = max(0,min(255,colors_copy[name]['G']))
        b = max(0,min(255,colors_copy[name]['B']))
        
        bbox[mask] = (b,g,r)
        
        # 사진에 텍스트로 레벨 결과 작성
        if name == 'white':
            temp = np.where(mask == True)
            x_1, y_1 = temp[0][-1], temp[1][-1]
            cv2.putText(bbox, str('Level:'), (y_1-90, x_1-60), cv2.FONT_HERSHEY_SIMPLEX, 2, (150,150,255), 5)
        else:
            level = levels[name]
            temp = np.where(mask == True)
            x_1, y_1 = temp[0][-1], temp[1][-1]
            cv2.putText(bbox, str(level), (y_1+20, x_1), cv2.FONT_HERSHEY_SIMPLEX, 2, (150,150,255), 5)
    
    # 사진 출력
    filename_split = filename.split('.')[0]
    cv2.imwrite(filename_split + '_mask.jpg', bbox)
    
    result['success'] = True
    
    return levels, result