import numpy as np
from sklearn.cluster import DBSCAN
import cv2

def cluster_and_get_medians(points, eps=8, min_samples=5):
    """
    주어진 좌표 리스트에서 DBSCAN 군집화를 수행하고 각 군집의 중앙값을 반환합니다.

    Args:
        points (list of tuple): [(x1, y1), (x2, y2), ...] 형태의 좌표 리스트
        eps (float): 군집화 반경 (좌표 간 거리 기준)
        min_samples (int): 군집을 이루는 최소 포인트 수

    Returns:
        list of tuple: 각 군집의 중앙값 [(median_x1, median_y1), ...]
    """
    # 1. numpy 배열로 변환
    points_np = np.array(points)

    # 2. DBSCAN 군집화
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np)

    # 각 좌표의 군집 레이블
    labels = dbscan.labels_

    # 군집별 좌표 분리
    unique_labels = set(labels)  # 군집 레이블 (노이즈는 -1)
    clusters = {label: [] for label in unique_labels if label != -1}  # 노이즈 제외
    for point, label in zip(points_np, labels):
        if label != -1:  # 노이즈는 제외
            clusters[label].append(point)

    # 각 군집의 중앙값 계산
    cluster_medians = []
    for cluster_points in clusters.values():
        cluster_points = np.array(cluster_points)
        median_x = int(np.median(cluster_points[:, 0]))  # x 좌표의 중앙값
        median_y = int(np.median(cluster_points[:, 1]))  # y 좌표의 중앙값
        cluster_medians.append((median_x, median_y))

    return cluster_medians  # 중앙값 좌표 반환


def filter_points_by_lines(points, line_threshold=2):
    """
    점들을 직선 기반으로 필터링합니다.

    Args:
        points (list of tuple): 점 좌표 리스트 [(x1, y1), (x2, y2), ...].
        line_threshold (float): 직선과 점 사이 허용 거리.

    Returns:
        list of tuple: 필터링된 점들.
    """
    points_np = np.array(points, dtype=np.int32)

    # 이미지 크기 설정
    max_x = max([p[0] for p in points]) + 1
    max_y = max([p[1] for p in points]) + 1
    blank_image = np.zeros((max_y, max_x), dtype=np.uint8)

    # 빨간 점들을 이미지에 표시
    for x, y in points:
        blank_image[y, x] = 255

    # 허프 변환으로 직선 검출
    lines = cv2.HoughLinesP(blank_image, 1, np.pi / 180, threshold=10, minLineLength=20, maxLineGap=5)

    # 직선 기반 점 필터링
    if lines is None:
        return points  # 검출된 직선이 없으면 원본 반환

    filtered_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        for x, y in points:
            # 점과 직선 간 거리 계산
            distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if distance < line_threshold:
                filtered_points.append((x, y))

    return filtered_points

if __name__ == "__main__":
    # 테스트용 데이터
    points = [
        (119, 326), (119, 332), (119, 354), (120, 329), (120, 354), (120, 360),
        (121, 328), (121, 354), (121, 360), (122, 327), (122, 354), (122, 363),
        (123, 328), (123, 354), (123, 364), (124, 327), (124, 354), (124, 364)
    ]

    # DBSCAN 실행 및 중앙값 출력
    cluster_medians = cluster_and_get_medians(points, eps=3, min_samples=5)
    print("DBSCAN 군집별 중앙값 좌표:", cluster_medians)
