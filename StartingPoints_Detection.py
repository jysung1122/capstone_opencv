import cv2
import numpy as np

# 그레이스케일 이미지 불러오기 (원본 이미지는 수정하지 않음)
#original_image = cv2.imread('difficult.png', cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread('difficult2.png', cv2.IMREAD_GRAYSCALE)
#original_image = cv2.imread('simple.png', cv2.IMREAD_GRAYSCALE)
#original_image = cv2.imread('simple2.png', cv2.IMREAD_GRAYSCALE)

# 내부적으로 밝기 값 40 이하를 0으로 처리
threshold_value = 45
_, thresholded_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_TOZERO)

# 원본 이미지를 컬러로 변환 (ROI 시각적 표시용)
image_with_roi = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

# 큰 ROI 설정
#x, y, w, h = 70, 30, 220, 350  # difficult1
x, y, w, h = 30, 30, 200, 350  # difficult2
#x, y, w, h = 30, 30, 220, 320  # simple.png
#x, y, w, h = 80, 10, 200, 500  # simple2

small_roi_width = int(2 * (w / 3))
small_roi_height = int(h / 8)
num_small_rois = 6  # 총 6개의 작은 ROI
small_roi_start_y = y + h // 2 - (small_roi_height * num_small_rois // 2)  # 작은 ROI 배치 시작점

# 첫 번째와 두 번째 점들의 좌표를 저장할 리스트
first_points = []
second_points = []

# 모든 작은 ROI에 대해 밝기 변화 감지 및 점 표시
for i in range(num_small_rois):
    small_roi_x = x + w - small_roi_width  # 오른쪽에 배치
    small_roi_y = small_roi_start_y + i * small_roi_height

    # 작은 ROI의 중간 y좌표 계산
    small_roi_middle_y = small_roi_y + small_roi_height // 2

    # 작은 ROI를 파란색 사각형으로 강조 표시
    #cv2.rectangle(image_with_roi, (small_roi_x, small_roi_y),
    #              (small_roi_x + small_roi_width, small_roi_y + small_roi_height), (255, 0, 0), 2)

    # 밝기 값 계산 및 급격한 변화 후 점 찍기
    previous_value = None
    delta_threshold = 5  # 급격한 변화 임계값
    points_marked = 0  # 표시된 점의 개수
    last_marked_position = None  # 마지막 점의 위치 저장

    for j in range(small_roi_width - 1, -1, -1):  # 오른쪽에서 왼쪽으로 스캔
        pixel_value = int(thresholded_image[small_roi_y, small_roi_x + j])  # 처리된 이미지 사용

        if previous_value is not None:
            delta = abs(pixel_value - previous_value)  # 정수형 연산으로 오버플로우 방지

            if delta >= delta_threshold and (points_marked < 2):
                if points_marked == 0 or abs(j - last_marked_position) > 5:
                    # 점 찍기
                    #cv2.circle(image_with_roi, (small_roi_x + j, small_roi_middle_y), 5, (0, 0, 255), -1)
                    points_marked += 1
                    last_marked_position = j  # 마지막 점의 위치 저장

                    # 첫 번째 점과 두 번째 점의 좌표를 각각 리스트에 저장
                    if points_marked == 1:
                        first_points.append((small_roi_x + j, small_roi_middle_y))
                    elif points_marked == 2:
                        second_points.append((small_roi_x + j, small_roi_middle_y))

        previous_value = pixel_value

# 각 리스트에서 x값들의 평균 계산
if first_points:
    first_x_mean = int(np.mean([p[0] for p in first_points]))
    # 첫 번째 점들의 평균 x값에 수직선 그리기
    cv2.line(image_with_roi, (first_x_mean, y), (first_x_mean, y + h), (0, 255, 255), 2)

if second_points:
    second_x_mean = int(np.mean([p[0] for p in second_points]))
    # 두 번째 점들의 평균 x값에 수직선 그리기
    cv2.line(image_with_roi, (second_x_mean, y), (second_x_mean, y + h), (255, 255, 0), 2)

# 큰 ROI를 초록색 사각형으로 강조 표시
cv2.rectangle(image_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 결과 시각화
cv2.imshow('ROI Visualization', image_with_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
