import cv2
import numpy as np

# 그레이스케일 이미지 불러오기
original_image = cv2.imread('difficult.png', cv2.IMREAD_GRAYSCALE)

# 내부적으로 밝기 값 45 이하를 0으로 처리
threshold_value = 45
_, thresholded_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_TOZERO)

# 원본 이미지를 컬러로 변환 (ROI 시각적 표시용)
image_with_roi = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

# 큰 ROI 설정
x, y, w, h = 70, 30, 220, 350  # ROI 좌표와 크기

# 큰 ROI를 초록색 사각형으로 강조 표시
cv2.rectangle(image_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 큰 ROI의 중간 y좌표 계산
big_roi_middle_y = y + h // 2

# 밝기 값 계산 및 급격한 변화 후 점 찍기
previous_value = None
delta_threshold = 5  # 급격한 변화 임계값
points_marked = 0  # 표시된 점의 개수
last_marked_position = None  # 마지막 점의 위치 저장

# 첫 번째 스캔: 오른쪽에서 왼쪽으로 스캔하며 점 찍기
for i in range(w - 1, -1, -1):  # 오른쪽에서 왼쪽으로 스캔
    pixel_value = int(thresholded_image[big_roi_middle_y, x + i])  # 처리된 이미지 사용

    if previous_value is not None:
        delta = abs(pixel_value - previous_value)  # 정수형 연산으로 오버플로우 방지
        if delta >= delta_threshold and (points_marked < 3):
            if points_marked == 0 or abs(i - last_marked_position) > 5:
                # 점 찍기
                cv2.circle(image_with_roi, (x + i, big_roi_middle_y), 3, (0, 0, 255), -1)
                points_marked += 1
                last_marked_position = i  # 마지막 점의 위치 저장

    previous_value = pixel_value

# 결과 시각화
cv2.imshow('ROI Visualization', image_with_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
