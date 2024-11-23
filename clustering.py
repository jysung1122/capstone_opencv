import numpy as np
from sklearn.cluster import DBSCAN

def cluster_and_get_medians(points, eps=3, min_samples=7):
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
