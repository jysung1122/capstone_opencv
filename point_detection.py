# point_detection.py

import cv2

def detect_edge_points_in_roi(thresholded_image, x, y, w, h, delta_threshold=5, max_points=2):
    """
    ROI 내부에서 밝기 변화 감지하여 점들을 추출합니다.

    Args:
        thresholded_image (numpy.ndarray): 전처리된 그레이스케일 이미지.
        x, y, w, h (int): ROI의 좌표와 크기.
        delta_threshold (int): 밝기 변화 감지 임계값.
        max_points (int): 각 행에서 감지할 최대 점 수.

    Returns:
        tuple: (first_points, second_points)
    """
    first_points = []
    second_points = []

    for i in range(y + 1, y + h - 1):  # ROI 내부의 y 좌표 (위에서 아래로)
        points_marked = 0
        previous_value = None
        last_marked_position = None

        for j in range(w - 1, -1, -1):  # ROI 내부의 x 좌표 (오른쪽에서 왼쪽으로)
            pixel_value = int(thresholded_image[i, x + j])

            if previous_value is not None:
                delta = abs(pixel_value - previous_value)
                if delta >= delta_threshold:
                    if points_marked == 0 or (points_marked < max_points and abs(j - last_marked_position) > 5):
                        if points_marked == 0:
                            first_points.append((x + j, i))
                        elif points_marked == 1:
                            second_points.append((x + j, i))

                        points_marked += 1
                        last_marked_position = j

                        if points_marked == max_points:
                            break

            previous_value = pixel_value

    return first_points, second_points

def detect_brightness_changes_in_roi(nlm_denoised, roi_x, roi_y, roi_w, roi_h, image_with_roi,
                                     delta_threshold=2, max_points=3, direction='up', point_color=(0, 0, 255)):
    """
    ROI에서 밝기 변화 지점을 탐지합니다.

    Args:
        nlm_denoised (numpy.ndarray): 노이즈 제거된 이미지.
        roi_x, roi_y, roi_w, roi_h (int): ROI의 좌표와 크기.
        image_with_roi (numpy.ndarray): 결과 시각화를 위한 이미지.
        delta_threshold (int): 밝기 변화 감지 임계값.
        max_points (int): 각 축에서 감지할 최대 점 수.
        direction (str): 스캔 방향 ('up' 또는 'down').
        point_color (tuple): 시각화할 점의 색상 (B, G, R).

    Returns:
        list: 변화 감지된 좌표들의 리스트.
    """
    points = []

    if direction == 'up':
        y_range = range(roi_y + roi_h - 1, roi_y, -1)
    else:
        y_range = range(roi_y, roi_y + roi_h)

    for j in range(roi_x, roi_x + roi_w):
        points_marked = 0
        previous_value = None
        last_marked_position = None

        for i in y_range:
            pixel_value = int(nlm_denoised[i, j])

            if previous_value is not None:
                delta = abs(pixel_value - previous_value)
                if delta >= delta_threshold:
                    if points_marked == 0 or (points_marked < max_points and abs(i - last_marked_position) > 5):
                        # 변화 감지 지점에 점 표시 (시각화가 필요 없으면 이 부분을 주석 처리하세요)
                        cv2.circle(image_with_roi, (j, i), 1, point_color, -1)

                        points.append((j, i))

                        points_marked += 1
                        last_marked_position = i

                        if points_marked == max_points:
                            break

            previous_value = pixel_value

    return points
