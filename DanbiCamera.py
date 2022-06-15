import numpy as np
import pandas as pd
import math
import cv2
    
# zoom
def camera_zoom(src1, src2):

    moving_array = []

    for i in range(1, 2335, 2): 
        src1 = cv2.imread(f'/notebooks/Danbi/Frame/{str(ANI)}/frame{i}.jpg', 0) # i 번째 원본 프레임
        src2 = cv2.imread(f'/notebooks/Danbi/camera_moving/new_sub_{str(ANI)}/subtraction{i + 3}.jpg', 0) # i+1 번째 원본과 셀리언시의 차영상 프레임
        src1 = cv2.GaussianBlur(src1,(15,15),0)
        src2 = cv2.GaussianBlur(src2,(15,15),0)
        src_pts, dst_pts, mask = cal_diff_AKAZE(src1, src2)
        if src_pts is None:
            continue

        N_match = len(np.unique(dst_pts * mask)) - 1
        N_correct = np.count_nonzero(mask) * 2
        pivot = np.abs(N_match - N_correct)

        if N_correct < 5 or pivot > 6:
            continue
        else:
            zoom = cal_zoom_var(src_pts, dst_pts, mask)

            if zoom == 'none':
                continue
            else:
                moving_array.append([f'{i}',  str(zoom)])

            
# zoom + tilting
def camera_zoom_tilt(src1, src2):

    moving_array = []

    for i in range(1, 2335, 2): 
        src1 = cv2.imread('frame{i}.jpg') # i 번째 원본 프레임
        src2 = cv2.imread('subtraction{i + 3}.jpg') # i+1 번째 원본과 셀리언시의 차영상 프레임
        src1 = cv2.GaussianBlur(src1,(15,15),0)
        src2 = cv2.GaussianBlur(src2,(15,15),0)
        src_pts, dst_pts, mask = cal_diff_AKAZE(src1, src2)
        if src_pts is None:
            continue

        N_match = len(np.unique(dst_pts * mask)) - 1
        N_correct = np.count_nonzero(mask) * 2
        pivot = np.abs(N_match - N_correct)

        if N_correct < 5 or pivot > 6:
            continue
        else:
            zoom = cal_zoom_var(src_pts, dst_pts, mask)
            trans = cal_trans(src_pts, dst_pts, mask)/np.count_nonzero(mask)
            total_trans = np.abs(trans[0]) + np.abs(trans[1])      
            for j in range(2):
                if np.abs(trans[j]) / total_trans < 0.05:
                    trans[j] = 0
                    zoom = 'none'

            X = np.where(trans[0] > 1, 'right', np.where(trans[0] < -1, 'left'
                                             , np.where(-1 < trans[0] < 1, 'none', trans[0])))
            Y = np.where(trans[1] > 1, 'up', np.where(trans[1] < -1, 'down'
                                              , np.where(-1 < trans[1] < 1, 'none', trans[1])))

            if X == 'none' and Y == 'none':
                pass
            else:
                zoom = zoom
                moving_array.append([f'{i}', str(X), str(Y), str(zoom)])


def cal_diff_AKAZE(src1, src2):
    feature = cv2.xfeatures2d.SIFT_create()

    # 특징점 검출 및 기술자 계산
    kp1, desc1 = feature.detectAndCompute(src1, None)
    kp2, desc2 = feature.detectAndCompute(src2, None)

    if desc1 is None:
        return None, None, None
    elif desc2 is None:
        return None, None, None

    matcher = cv2.BFMatcher()
    matches = matcher.match(desc1, desc2)

    # 매칭 결과
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:100]

    # queryIdx원본 영상의 좌표
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
    # trainIdx대상 영상의 좌표
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
    if len(src_pts) <= 3 or len(dst_pts) <= 3:
        return None, None, None
    else:
    # 원근 변환 행렬
        mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return src_pts, dst_pts, mask
    

def cal_zoom_var(src_pts, dst_pts, mask):
    true_src = src_pts * mask
    true_src_list = true_src[true_src>0].reshape(-1, 2)
    true_dst = dst_pts * mask
    true_dst_list = true_dst[true_dst>0].reshape(-1, 2)
    N_true = len(mask[mask>0])

    result_fin_src = 0
    result_fin_dst = 0
    for i in range(N_true):
        if i == N_true - 1:
            break
        sub_src = true_src_list[i] - true_src_list[i + 1]
        dou_src = sub_src**2
        fin_src = np.sum(dou_src)
        fin_src = np.sqrt(fin_src)
        result_fin_src += fin_src
        sub_dst = true_dst_list[i] - true_dst_list[i + 1]
        dou_dst = sub_dst**2
        fin_dst = np.sum(dou_dst)
        fin_dst = np.sqrt(fin_dst)
        result_fin_dst += fin_dst

    ratio = result_fin_dst / result_fin_src
    
    if ratio > 1.01:
        zoom = 'in'
    elif ratio < 0.99:
        zoom = 'out'
    else:
        zoom = 'none'

    return zoom


def cal_trans(src_pts, dst_pts, mask):
    src_mask = src_pts * mask
    dst_mask = dst_pts * mask
    translation = np.sum(src_mask - dst_mask, axis = 0)
    return translation


def new_detect_object_box(src):
  
    if src is None:
        print('Image load failed!')
        sys.exit()

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    new_stats = []

    for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
        (x, y, w, h, area) = stats[i]

      # 노이즈 제거
        if area < 15000 or area > 400000 or (w*h)/(stats[0][2]*stats[0][3])> 0.15:
            continue
        new_stats.append(stats[i])

    new_stats = np.array(new_stats)
    if len(new_stats) != 0:
        new_stats = new_stats[new_stats[:, 0].argsort()[::-1]]

    return new_stats

def center_coor(stats):
    center_x = (stats[:, 0] + stats[:, 2])/2
    center_y = (stats[:, 1] + stats[:, 3])/2
    center_coor = np.vstack((center_x, center_y)).T
    return center_coor

def new_detect_object_box(src):
  
    if src is None:
        print('Image load failed!')
        sys.exit()

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    new_stats = []

    for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
        (x, y, w, h, area) = stats[i]

      # 노이즈 제거
        if area < 15000 or area > 400000 or (w*h)/(stats[0][2]*stats[0][3])> 0.15:
            continue
        new_stats.append(stats[i])

    new_stats = np.array(new_stats)
    if len(new_stats) != 0:
        new_stats = new_stats[new_stats[:, 0].argsort()[::-1]]

    return new_stats

def center_coor(stats):
    center_x = (stats[:, 0] + stats[:, 2]/2)
    center_y = (stats[:, 1] + stats[:, 3]/2)
    center_coor = np.vstack((center_x, center_y)).T
    return center_coor

def cal_rad(arr):
    rad = math.atan2(arr[1] - arr[3], arr[0] - arr[2])
    deg = (rad * 180) / math.pi
    if deg < 0:
        deg += 180
    if deg < 30:
        deg = 0
    elif deg < 60:
        deg = 30
    elif deg < 90:
        deg = 60
    elif deg < 120:
        deg = 90
    elif deg < 150:
        deg = 120
    elif deg < 180:
        deg = 150       
    return int(deg)


def object_tracking_main(resource_path,count):
    tracking_list_ = []

    for i in range(0, count - 2): 

        src = cv2.imread(resource_path+'HIT/diff%d.jpg' % i, 0)
        src_g = cv2.GaussianBlur(src,(3,3),0)

        # 좌표가 잡히지 않는다면 continue
        stats = np.array(new_detect_object_box(src_g)) + 1
        if len(stats) == 0:
            continue

        ctr_coor = center_coor(stats)
        new_s, new_cc = del_overlap(ctr_coor, stats)

        if len(new_cc) >= 5:
            continue

        # 처음 시작하는 단계이거나 혹은 다른 씬으로 바뀌면서 프레임 수가 연속되지 않으면 시작 좌표 추가
        if len(tracking_list_) == 0 or i - tracking_list_[len(tracking_list_)- 1][0] >= 2:
            start_object_box = [0] * 30
            start_object_box[0] = i
            for stats_idx in range(len(new_cc)):
                start_object_box[stats_idx + 1] = tuple(new_cc[stats_idx])
            tracking_list_ += [start_object_box]
            continue

        # object_box 에서 지난 점 중 0이 아닌 점의 index 구하기
        past_index = []
        for j in range(1, len(tracking_list_[len(tracking_list_)- 1])):
            if tracking_list_[len(tracking_list_)- 1][j] != 0:
                past_index += [j]

        # 이전 점과 다음 점 사이의 거리 구하기
        dist_list_ = []
        for p in past_index:
            for n in range(len(new_cc)):
                dist = np.linalg.norm(np.array(tracking_list_[len(tracking_list_)-1][p]) - new_cc[n])
                dist_list_.append(dist)
        dist_array_ = np.array(dist_list_)
        re_dist_array_ = dist_array_.reshape(-1, len(new_cc))

        # 이전 점과 가장 가까운 다음 점의 인덱스 연결
        index_list = []
        for r in range(len(re_dist_array_)):
            minindex = np.argmin(re_dist_array_[r])
            if re_dist_array_[r, minindex] < 130 :
                index_list.append([past_index[r], minindex])
        if len(index_list) == 0:
            continue

        # 다음 점 추가 할 array 생성 및 삭제 리스트 생성
        del_list = []
        next_object_box = [0] * 30
        for d in range(len(index_list)):
            next_object_box[0] = i
            next_object_box[index_list[d][0]] = tuple(new_cc[index_list[d][1]])
            del_list.append(index_list[d][1])

        # 이전 오브젝트를 따라간 후 남은 오브젝트 추가
        rest_list = np.delete(new_cc, del_list, axis = 0)
        if len(rest_list) != 0:
            for e in range(len(rest_list)):
                next_object_box[max(index_list)[0] + e + 1] = tuple(rest_list[e])    

        tracking_list_ += [next_object_box]
        
    return make_list(tracking_list_)