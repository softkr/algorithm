import numpy as np

import os, getopt, sys, time
import json

from tqdm import tqdm

import urine_strip as us
from urine_config import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

bbox = us.BlackBox(bbox_points, roi_size, qr_centers, qr_size, min_area_ratio=0.01)
strip = us.Strip(bbox_points, patch_points, patch_hsize, ref_points, strip_length, bbox_padding)


def estimate(filename):
    result = {
        'success': False
    }
    print('\n')
    # black_box.py 실행
    try:
        result_bbox = bbox(filename)
        return
    except:
        result['error'] = "UNKNOWN"
        return result

    if not result_bbox['success']:
        result['error'] = "QR_BBOX"
        return result

    # strip.py 실행 (스트립 인식 후 색상 추출)
    try:
        result_strip = strip(result_bbox['bbox'], filename, result_bbox['roi'])
    except:
        result['error'] = "UNKNOWN"
        return result

    if not result_strip['success']:
        if 'error' in result_strip.keys():
            result['error'] = result_strip['error']
        else:
            result['error'] = 'STRIP'
        return result

    # estimate.py 실행 (레벨 예측)
    try:
        levels, result_est = us.estimate(model_ref_RGB, result_strip['colors'], result_strip['out'],
                                         result_strip['masks'], filename)
    except:
        result['error'] = "UNKNOWN"
        return result

    if not result_est['success']:
        result['error'] = result_est['error']
        return result

    # 결과 저장
    result['success'] = True
    result['levels'] = levels
    result['colors'] = result_strip['colors']

    return result


def main(argv, report):
    FILENAME = argv[0]
    INFILE = ''
    OUTFILE = 'estimate.csv'
    try:
        opts, etc_args = getopt.getopt(argv[1:], 'i:o:', ['infile', 'out'])
    except getopt.GetoptError:
        print(FILENAME, '-i <infile> [-o <outdir>]')
        sys.exit(-1)

    for opt, arg in opts:
        if opt in ('-i', '--infile'):
            INFILE = arg
        elif opt in ('-o', '--out'):
            OUTFILE = arg

    report['total'] = 0
    report['success'] = 0
    report['analysis_time'] = analysis_time = []

    with open(OUTFILE, 'w') as fp:
        files = us.glob(INFILE)
        pbar: tqdm[str | bytes] = tqdm(files)

        fp.write('filename, error')
        fp.write('\n')

        for filename in pbar:
            report['total'] += 1

            start_time = time.time()
            result = estimate(filename)
            return result

            success = result['success']
            error = result.get('error')
            fp.write(f'{filename}, {error}')

            # print(json.dumps(result))
            analysis_time.append(time.time() - start_time)

            if success:
                report['success'] += 1
                msg = 'PASS'

                levels = result['levels']
                fp.write('\ncategory, R, G, B, STD, estimated level\n')

                for category in result['colors']:
                    R = round(result['colors'][category]['R'], 2)
                    G = round(result['colors'][category]['G'], 2)
                    B = round(result['colors'][category]['B'], 2)
                    stdev = round(result['colors'][category]['std'], 2)

                    row = f'{category},{R},{G},{B},{stdev}'

                    if category in ['blood', 'ketone', 'protein', 'glucose', 'pH']:
                        row += f',{levels[category]}'

                    fp.write(row + '\n')

            else:
                msg = 'FAIL'

            fn = filename
            if len(fn) > 40:
                fn = fn[-40:]
            pbar.set_description(f'{fn:<40}    {msg}')


if __name__ == "__main__":
    report = {
        'analysis_time': [],
        'total': 0,
        'success': 0
    }
    try:
        main(sys.argv, report)
    except KeyboardInterrupt:
        pass

    analysis_time = np.array(report['analysis_time'])
