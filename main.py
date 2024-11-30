import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from clustering import *
from point_detection import *

# 그레이스케일 이미지 불러오기
#original_image = cv2.imread('difficult.png', cv2.IMREAD_GRAYSCALE)
#original_image = cv2.imread('difficult2.png', cv2.IMREAD_GRAYSCALE)
#original_image = cv2.imread('simple.png', cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread('simple2.png', cv2.IMREAD_GRAYSCALE)

# 큰 ROI 설정
#x, y, w, h = 70, 30, 220, 350  # difficult1
#x, y, w, h = 30, 30, 200, 370  # difficult2
#x, y, w, h = 30, 30, 220, 320  # simple.png
x, y, w, h = 80, 10, 200, 500  # simple2

# ROI 시각적 표시를 위한 컬러 변환
image_with_roi = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

# ROI 강조 표시
cv2.rectangle(image_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 원본 이미지 전체에 밝기 값 45 이하를 0으로 처리
threshold_value = 45
_, thresholded_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_TOZERO)


# ROI 내부에서 밝기 변화 감지하여 점 찍기
first_points, second_points = detect_edge_points_in_roi(
    thresholded_image, x, y, w, h, delta_threshold=5, max_points=2
)

# # first_points 시각화
# for point in first_points:
#     x_coord, y_coord = point
#     cv2.circle(image_with_roi, (x_coord, y_coord), 2, (0, 0, 255), -1)
#
# # second_points 시각화
# for point in second_points:
#     x_coord, y_coord = point
#     cv2.circle(image_with_roi, (x_coord, y_coord), 2, (0, 0, 255), -1)


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

# 이미지 크기
image_width, image_height = original_image.shape[1], original_image.shape[0]

# 왼쪽 위 작은 ROI
top_left_roi_x = max(0, second_median[0] - 85)
top_left_roi_y = max(0, y)
top_left_roi_w = min(80, image_width - top_left_roi_x)
top_left_roi_h = max(0, y + h - min(first_median[1], second_median[1]) - 120)

# 오른쪽 위 작은 ROI
top_right_roi_x = max(0, second_median[0] + 5)
top_right_roi_y = max(0, y)
top_right_roi_w = max(0, first_median[0] - top_right_roi_x - 5)
top_right_roi_h = max(0, y + h - min(first_median[1], second_median[1]) - 120)

# 왼쪽 아래 작은 ROI
btm_left_roi_x = max(0, second_median[0] - 85)
btm_left_roi_y = max(0, int(max(first_median[1], second_median[1]) + 120))
btm_left_roi_w = min(80, image_width - btm_left_roi_x)
btm_left_roi_h = max(0, y + h - btm_left_roi_y)

# 오른쪽 아래 작은 ROI
btm_right_roi_x = max(0, second_median[0] + 5)
btm_right_roi_y = max(0, int(max(first_median[1], second_median[1]) + 120))
btm_right_roi_w = max(0, first_median[0] - btm_right_roi_x - 5)
btm_right_roi_h = max(0, y + h - btm_right_roi_y)

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
top_left_points = detect_brightness_changes_in_roi(
    nlm_denoised, top_left_roi_x, top_left_roi_y, top_left_roi_w, top_left_roi_h,
    image_with_roi, delta_threshold=2, max_points=3, direction='up', point_color=(0, 0, 255)
)

# 직선 기반 점 필터링
filtered_points = filter_points_by_lines(top_left_points, line_threshold=2)

# DBSCAN 군집 중앙값 계산
top_left_cluster_medians = cluster_and_get_medians(filtered_points)

# 중앙값 좌표에 동그라미 그리기
# for (x, y) in top_left_cluster_medians:
#     cv2.circle(image_with_roi, (x, y), 3, (255, 255, 0), -1)  # 초록색 동그라미


# 오른쪽 위 작은 ROI 밑에서부터 위로 스캔하기 (y축 기준으로 밝기 변화 탐지)
top_right_points = detect_brightness_changes_in_roi(
    nlm_denoised, top_right_roi_x, top_right_roi_y, top_right_roi_w, top_right_roi_h,
    image_with_roi, delta_threshold=2, max_points=3, direction='up', point_color=(0, 0, 255)
)

# 직선 기반 점 필터링
filtered_points = filter_points_by_lines(top_right_points, line_threshold=2)

# DBSCAN 군집 중앙값 계산
top_right_cluster_medians = cluster_and_get_medians(filtered_points)
# 중앙값 좌표에 동그라미 그리기
# for (x, y) in top_right_cluster_medians:
#     cv2.circle(image_with_roi, (x, y), 3, (255, 255, 0), -1)  # 초록색 동그라미

# 왼쪽 아래 작은 ROI 위에서부터 밑으로 스캔하기 (y축 기준으로 밝기 변화 탐지)
btm_left_points = detect_brightness_changes_in_roi(
    nlm_denoised, btm_left_roi_x, btm_left_roi_y, btm_left_roi_w, btm_left_roi_h,
    image_with_roi, delta_threshold=2, max_points=3, direction='down', point_color=(0, 0, 255)
)

# 직선 기반 점 필터링
filtered_points = filter_points_by_lines(btm_left_points, line_threshold=2)

# DBSCAN 군집 중앙값 계산
btm_left_cluster_medians = cluster_and_get_medians(filtered_points)
# 중앙값 좌표에 동그라미 그리기
# for (x, y) in btm_left_cluster_medians:
#     cv2.circle(image_with_roi, (x, y), 3, (255, 255, 0), -1)  # 초록색 동그라미


# 오른쪽 아래 작은 ROI 위에서부터 밑으로 스캔하기 (y축 기준으로 밝기 변화 탐지)
btm_right_points = detect_brightness_changes_in_roi(
    nlm_denoised, btm_right_roi_x, btm_right_roi_y, btm_right_roi_w, btm_right_roi_h,
    image_with_roi, delta_threshold=2, max_points=3, direction='down', point_color=(0, 0, 255)
)

# 직선 기반 점 필터링
filtered_points = filter_points_by_lines(btm_right_points, line_threshold=2)

# DBSCAN 군집 중앙값 계산
btm_right_cluster_medians = cluster_and_get_medians(filtered_points)
# 중앙값 좌표에 동그라미 그리기
# for (x, y) in btm_right_cluster_medians:
#     cv2.circle(image_with_roi, (x, y), 3, (255, 255, 0), -1)  # 초록색 동그라미


# y좌표가 가장 큰 점 선택
if top_left_cluster_medians:
    top_left_point = max(top_left_cluster_medians, key=lambda p: p[1])
    #print("Top Left Point:", top_left_point)
else:
    top_left_point = None

# y좌표가 가장 큰 점 선택
if top_right_cluster_medians:
    top_right_point = max(top_right_cluster_medians, key=lambda p: p[1])
    #print("Top Right Point:", top_right_point)
else:
    top_right_point = None

# y좌표가 가장 작은 점 선택
if btm_left_cluster_medians:
    btm_left_point = min(btm_left_cluster_medians, key=lambda p: p[1])
    #print("Bottom Left Point:", btm_left_point)
else:
    btm_left_point = None

# y좌표가 가장 작은 점 선택
if btm_right_cluster_medians:
    btm_right_point = min(btm_right_cluster_medians, key=lambda p: p[1])
    #print("Bottom Right Point:", btm_right_point)
else:
    btm_right_point = None

if top_left_point is not None:
    cv2.circle(image_with_roi, top_left_point, 3, (0, 255, 0), -1)  # 초록색 점

if top_right_point is not None:
    cv2.circle(image_with_roi, top_right_point, 3, (0, 255, 0), -1)

if btm_left_point is not None:
    cv2.circle(image_with_roi, btm_left_point, 3, (0, 255, 0), -1)

if btm_right_point is not None:
    cv2.circle(image_with_roi, btm_right_point, 3, (0, 255, 0), -1)

overlap_top_peak = max(top_left_point[1], top_right_point[1])
overlap_btm_peak = min(btm_left_point[1], btm_right_point[1])

#print("Top Peak:", overlap_top_peak)
#print("Bottom Peak:", overlap_btm_peak)

# top_peak에 20을 더하고, btm_peak에서 20을 뺌
overlap_top_peak += 20
overlap_btm_peak -= 20

# welding_points 리스트에 점들을 추가
welding_points = []
for point in second_points:
    x_coord, y_coord = point
    if overlap_top_peak <= y_coord <= overlap_btm_peak:
        welding_points.append(point)
        #cv2.circle(image_with_roi, (x_coord, y_coord), 1, (255, 0, 255), -1)

# x좌표 기준으로 필터링: x좌표가 second_median의 x좌표와 ±5 이상 차이나면 제거
filtered_welding_points = [
    point for point in welding_points if abs(point[0] - second_median[0]) <= 5
]

# welding_points를 y좌표 기준으로 정렬
welding_points_sorted = sorted(filtered_welding_points, key=lambda p: p[1])

# 점들을 연결하는 선 그리기
for i in range(len(welding_points_sorted) - 1):
    pt1 = welding_points_sorted[i]
    pt2 = welding_points_sorted[i + 1]
    cv2.line(image_with_roi, pt1, pt2, (255, 0, 255), 2)  # 보라색 선으로 연결


# 결과 시각화
cv2.imshow('ROI Visualization', image_with_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
