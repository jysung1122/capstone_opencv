import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from clustering import *

# 그레이스케일 이미지 불러오기
original_image = cv2.imread('difficult.png', cv2.IMREAD_GRAYSCALE)
#original_image = cv2.imread('difficult2.png', cv2.IMREAD_GRAYSCALE)
#original_image = cv2.imread('simple.png', cv2.IMREAD_GRAYSCALE)
#original_image = cv2.imread('simple2.png', cv2.IMREAD_GRAYSCALE)

# 큰 ROI 설정
x, y, w, h = 70, 30, 220, 350  # difficult1
#x, y, w, h = 30, 30, 200, 370  # difficult2
#x, y, w, h = 30, 30, 220, 320  # simple.png
#x, y, w, h = 80, 10, 200, 500  # simple2

# ROI 시각적 표시를 위한 컬러 변환
image_with_roi = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

# ROI 강조 표시
cv2.rectangle(image_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 원본 이미지 전체에 밝기 값 45 이하를 0으로 처리
threshold_value = 45
_, thresholded_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_TOZERO)

# 첫 번째와 두 번째로 찍힌 점들의 리스트 초기화
first_points = []
second_points = []

# ROI 내부에서 밝기 변화 감지하여 점 찍기
for i in range(y + 1, y + h - 1):  # ROI 내부의 y 좌표 (위에서 아래로)
    points_marked = 0  # 매 행(row)마다 points_marked를 초기화
    previous_value = None  # 각 행(row)마다 previous_value 초기화
    last_marked_position = None

    for j in range(w - 1, -1, -1):  # ROI 내부의 x 좌표 (오른쪽에서 왼쪽으로)
        pixel_value = int(thresholded_image[i, x + j])  # 현재 (x + j, i) 위치의 픽셀 값

        if previous_value is not None:
            delta = abs(pixel_value - previous_value)  # 이전 픽셀과의 밝기 차이 계산
            if delta >= 5:
                if points_marked == 0 or (points_marked < 2 and abs(j - last_marked_position) > 5):
                    # 변화 감지 지점에 빨간 점 표시
                    #cv2.circle(image_with_roi, (x + j, i), 1, (0, 0, 255), -1)

                    # 첫 번째와 두 번째 점에 따라 리스트에 추가
                    if points_marked == 0:
                        first_points.append((x + j, i))  # 첫 번째 점 리스트에 추가
                    elif points_marked == 1:
                        second_points.append((x + j, i))  # 두 번째 점 리스트에 추가

                    points_marked += 1  # 변화 지점 카운트 증가
                    last_marked_position = j

                    # 두 개의 변화 지점을 찾았으면 더 이상 점 찍기 중지
                    if points_marked == 2:
                        break

        previous_value = pixel_value  # 이전 픽셀 값 업데이트

# 각 행마다 첫 번째 점들과 두 번째 점들의 리스트 출력
# print("첫 번째 점들의 리스트:", first_points)
# print("두 번째 점들의 리스트:", second_points)

# 첫 번째 점들의 x, y 좌표 추출
first_x = [point[0] for point in first_points]
first_y = [point[1] for point in first_points]

# 두 번째 점들의 x, y 좌표 추출
second_x = [point[0] for point in second_points]
second_y = [point[1] for point in second_points]

# 중앙값 계산 및 정수형으로 변환
first_median = (int(np.median(first_x)), int(np.median(first_y)))
second_median = (int(np.median(second_x)), int(np.median(second_y)))

# 결과 출력
# print("첫 번째 점들의 중앙값:", first_median)
# print("두 번째 점들의 중앙값:", second_median)

# 중앙값 위치에 노란 점 표시
cv2.circle(image_with_roi, first_median, 3, (0, 255, 255), -1)  # 첫 번째 중앙값 좌표에 노란 점
cv2.circle(image_with_roi, second_median, 3, (0, 255, 255), -1)  # 두 번째 중앙값 좌표에 노란 점

# 왼쪽 위 작은 ROI 위치 및 크기 계산
top_left_roi_x = second_median[0] - 85
top_left_roi_y = y
top_left_roi_w = 80
top_left_roi_h = y + h - min(first_median[1], second_median[1]) - 120

# 오른쪽 위 작은 ROI 위치 및 크기 계산
top_right_roi_x = second_median[0] + 5
top_right_roi_y = y
top_right_roi_w = first_median[0] - top_right_roi_x - 5
top_right_roi_h = y + h - min(first_median[1], second_median[1]) - 120

# 왼쪽 아래 작은 ROI 위치 및 크기 계산
btm_left_roi_x = second_median[0] - 85
btm_left_roi_y = int(max(first_median[1], second_median[1]) + 120)
btm_left_roi_w = 80
btm_left_roi_h = y + h - max(first_median[1], second_median[1]) - 120

# 오른쪽 아래 작은 ROI 위치 및 크기 계산
btm_right_roi_x = second_median[0] + 5
btm_right_roi_y = int(max(first_median[1], second_median[1]) + 120)
btm_right_roi_w = first_median[0] - btm_right_roi_x - 5
btm_right_roi_h = y + h - max(first_median[1], second_median[1]) - 120

# 왼쪽 위 작은 ROI 그리기
cv2.rectangle(image_with_roi, (top_left_roi_x, top_left_roi_y),
              (top_left_roi_x + top_left_roi_w, top_left_roi_y + top_left_roi_h),
              (255, 0, 0), 2)

# 오른쪽 위 작은 ROI 그리기
cv2.rectangle(image_with_roi, (top_right_roi_x, top_right_roi_y),
              (top_right_roi_x + top_right_roi_w, top_right_roi_y + top_right_roi_h),
              (255, 0, 0), 2)

# 왼쪽 아래 작은 ROI 작은 ROI 그리기
cv2.rectangle(image_with_roi, (btm_left_roi_x, btm_left_roi_y),
              (btm_left_roi_x + btm_left_roi_w, btm_left_roi_y + btm_left_roi_h),
              (255, 0, 0), 2)

# 오른쪽 아래 작은 ROI 그리기
cv2.rectangle(image_with_roi, (btm_right_roi_x, btm_right_roi_y),
              (btm_right_roi_x + btm_right_roi_w, btm_right_roi_y + btm_right_roi_h),
              (255, 0, 0), 2)

# Non-Local Means 필터 적용
nlm_denoised = cv2.fastNlMeansDenoising(
    thresholded_image, None, h=10, templateWindowSize=7, searchWindowSize=21
)

# 왼쪽 위 작은 ROI 밑에서부터 위로 스캔하기 (y축 기준으로 밝기 변화 탐지)
top_left_points = []  # 변화 감지 좌표를 저장할 리스트 생성
for j in range(top_left_roi_x, top_left_roi_x + top_left_roi_w):  # 왼쪽에서 오른쪽으로 x 좌표 스캔
    points_marked = 0  # 각 x축에 대해 points_marked를 초기화
    previous_value = None  # 이전 밝기 값 초기화
    last_marked_position = None

    for i in range(top_left_roi_y + top_left_roi_h - 1, top_left_roi_y, -1):  # 밑에서부터 위로 y 좌표 스캔
        pixel_value = int(nlm_denoised[i, j])  # 현재 (j, i) 위치의 픽셀 값

        if previous_value is not None:
            delta = abs(pixel_value - previous_value)  # 이전 픽셀과의 밝기 차이 계산
            if delta >= 2:  # 밝기 변화가 2 이상이면 변화 감지로 판단
                if points_marked == 0 or (points_marked < 3 and abs(i - last_marked_position) > 5):
                    # 변화 감지 지점에 빨간 점 표시
                    cv2.circle(image_with_roi, (j, i), 1, (0, 0, 255), -1)

                    # 변화 감지된 좌표를 리스트에 추가
                    top_left_points.append((j, i))

                    points_marked += 1  # 변화 지점 카운트 증가
                    last_marked_position = i  # 현재 y 좌표를 마지막 마킹 위치로 저장

                    # 세 개의 변화 지점을 찾았으면 더 이상 점 찍기 중지
                    if points_marked == 3:
                        break

        previous_value = pixel_value  # 현재 밝기 값을 이전 값으로 업데이트

# DBSCAN 군집 중앙값 계산
top_left_cluster_medians = cluster_and_get_medians(top_left_points)
# 중앙값 좌표에 동그라미 그리기
for (x, y) in top_left_cluster_medians:
    cv2.circle(image_with_roi, (x, y), 3, (255, 255, 0), -1)  # 초록색 동그라미


top_right_points = []  # 변화 감지 좌표를 저장할 리스트 생성
# 오른쪽 위 작은 ROI 밑에서부터 위로 스캔하기 (y축 기준으로 밝기 변화 탐지)
for j in range(top_right_roi_x, top_right_roi_x + top_right_roi_w):  # 왼쪽에서 오른쪽으로 x 좌표 스캔
    points_marked = 0  # 각 x축에 대해 points_marked를 초기화
    previous_value = None  # 이전 밝기 값 초기화
    last_marked_position = None

    for i in range(top_right_roi_y + top_right_roi_h - 1, top_right_roi_y, -1):  # 밑에서부터 위로 y 좌표 스캔
        pixel_value = int(nlm_denoised[i, j])  # 현재 (j, i) 위치의 픽셀 값

        if previous_value is not None:
            delta = abs(pixel_value - previous_value)  # 이전 픽셀과의 밝기 차이 계산
            if delta >= 2:  # 밝기 변화가 2 이상이면 변화 감지로 판단
                if points_marked == 0 or (points_marked < 3 and abs(i - last_marked_position) > 5):
                    # 변화 감지 지점에 빨간 점 표시
                    cv2.circle(image_with_roi, (j, i), 1, (0, 0, 255), -1)

                    top_right_points.append((j, i))

                    points_marked += 1  # 변화 지점 카운트 증가
                    last_marked_position = i  # 현재 y 좌표를 마지막 마킹 위치로 저장

                    # 두 개의 변화 지점을 찾았으면 더 이상 점 찍기 중지
                    if points_marked == 3:
                        break

        previous_value = pixel_value  # 현재 밝기 값을 이전 값으로 업데이트


# DBSCAN 군집 중앙값 계산
top_right_cluster_medians = cluster_and_get_medians(top_right_points)
# 중앙값 좌표에 동그라미 그리기
for (x, y) in top_right_cluster_medians:
    cv2.circle(image_with_roi, (x, y), 3, (255, 255, 0), -1)  # 초록색 동그라미


btm_left_points = []  # 변화 감지 좌표를 저장할 리스트 생성
# 왼쪽 아래 작은 ROI 위에서부터 밑으로 스캔하기 (y축 기준으로 밝기 변화 탐지)
for j in range(btm_left_roi_x, btm_left_roi_x + btm_left_roi_w):  # 왼쪽에서 오른쪽으로 x 좌표 스캔
    points_marked = 0  # 각 x축에 대해 points_marked를 초기화
    previous_value = None  # 이전 밝기 값 초기화
    last_marked_position = None

    for i in range(btm_left_roi_y, btm_left_roi_y + btm_left_roi_h):  # 위에서부터 아래로 y 좌표 스캔
        pixel_value = int(nlm_denoised[i, j])  # 현재 (j, i) 위치의 픽셀 값

        if previous_value is not None:
            delta = abs(pixel_value - previous_value)  # 이전 픽셀과의 밝기 차이 계산
            if delta >= 2:  # 밝기 변화가 2 이상이면 변화 감지로 판단
                if points_marked == 0 or (points_marked < 3 and abs(i - last_marked_position) > 5):
                    # 변화 감지 지점에 빨간 점 표시
                    cv2.circle(image_with_roi, (j, i), 1, (0, 0, 255), -1)

                    btm_left_points.append((j, i))

                    points_marked += 1  # 변화 지점 카운트 증가
                    last_marked_position = i  # 현재 y 좌표를 마지막 마킹 위치로 저장

                    # 세 개의 변화 지점을 찾았으면 더 이상 점 찍기 중지
                    if points_marked == 3:
                        break

        previous_value = pixel_value  # 현재 밝기 값을 이전 값으로 업데이트

# DBSCAN 군집 중앙값 계산
btm_left_cluster_medians = cluster_and_get_medians(btm_left_points)
# 중앙값 좌표에 동그라미 그리기
for (x, y) in btm_left_cluster_medians:
    cv2.circle(image_with_roi, (x, y), 3, (255, 255, 0), -1)  # 초록색 동그라미


btm_right_points = []
# 오른쪽 아래 작은 ROI 위에서부터 밑으로 스캔하기 (y축 기준으로 밝기 변화 탐지)
for j in range(btm_right_roi_x, btm_right_roi_x + btm_right_roi_w):  # 왼쪽에서 오른쪽으로 x 좌표 스캔
    points_marked = 0  # 각 x축에 대해 points_marked를 초기화
    previous_value = None  # 이전 밝기 값 초기화
    last_marked_position = None

    for i in range(btm_right_roi_y, btm_right_roi_y + btm_right_roi_h):  # 위에서부터 아래로 y 좌표 스캔
        pixel_value = int(nlm_denoised[i, j])  # 현재 (j, i) 위치의 픽셀 값

        if previous_value is not None:
            delta = abs(pixel_value - previous_value)  # 이전 픽셀과의 밝기 차이 계산
            if delta >= 2:  # 밝기 변화가 2 이상이면 변화 감지로 판단
                if points_marked == 0 or (points_marked < 3 and abs(i - last_marked_position) > 5):
                    # 변화 감지 지점에 빨간 점 표시
                    cv2.circle(image_with_roi, (j, i), 1, (0, 0, 255), -1)

                    btm_right_points.append((j, i))

                    points_marked += 1  # 변화 지점 카운트 증가
                    last_marked_position = i  # 현재 y 좌표를 마지막 마킹 위치로 저장

                    # 세 개의 변화 지점을 찾았으면 더 이상 점 찍기 중지
                    if points_marked == 3:
                        break

        previous_value = pixel_value  # 현재 밝기 값을 이전 값으로 업데이트


# DBSCAN 군집 중앙값 계산
btm_right_cluster_medians = cluster_and_get_medians(btm_right_points)
# 중앙값 좌표에 동그라미 그리기
for (x, y) in btm_right_cluster_medians:
    cv2.circle(image_with_roi, (x, y), 3, (255, 255, 0), -1)  # 초록색 동그라미



# 결과 시각화
cv2.imshow('ROI Visualization', image_with_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
