import cv2
import numpy as np

# 그레이스케일 이미지 불러오기 (원본 이미지는 수정하지 않음)
original_image = cv2.imread('difficult.png', cv2.IMREAD_GRAYSCALE)
#original_image = cv2.imread('simple.png', cv2.IMREAD_GRAYSCALE)

# Non-Local Means 필터 적용 (필터 강도 및 범위 조정)
nlm_denoised = cv2.fastNlMeansDenoising(
    original_image,
    None,
    h=10,  # 필터 강도 증가 (기본값: 10)
    templateWindowSize=7,  # 템플릿 크기 증가 (기본값: 7)
    searchWindowSize=21  # 탐색 범위 증가 (기본값: 21)
)
# 내부적으로 밝기 값 45 이하를 0으로 처리
threshold_value = 45
_, thresholded_image = cv2.threshold(nlm_denoised, threshold_value, 255, cv2.THRESH_TOZERO)

# 원본 이미지를 컬러로 변환 (ROI 시각적 표시용)
image_with_roi = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

# 결과 시각화
cv2.imshow('ROI Visualization', image_with_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
