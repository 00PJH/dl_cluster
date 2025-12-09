import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수 (이 값을 조절해서 겉 테두리를 날리세요!)
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\ 8_clahe_hsv2'

# [핵심 1] 톱니바퀴 산 검출을 위한 검색 범위 (반지름 픽셀)
# 이 값보다 바깥쪽(겉 테두리 포함)은 아예 무시합니다.
# 이미지를 1300x1300으로 크롭하므로, 반지름은 최대 650입니다.
# 겉 테두리가 보통 550~600 부근에 있다면, 이를 500~520 정도로 제한하세요.
OUTER_LIMIT_RADIUS = 540

# 이 값보다 안쪽(센터홀)은 무시합니다.
INNER_LIMIT_RADIUS = 350

# 톱니(산)로 인정할 최소 간격
MIN_PEAK_DISTANCE = 15

# 크롭 크기
CROP_SIZE = 1500

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def imread_korean(file_path):
    try:
        img_array = np.fromfile(file_path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_korean(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        return False
    except Exception:
        return False

# ==========================================
# 3. 중심 잡기 및 크롭 (V10 로직 재사용 - 가장 안정적)
# ==========================================
def get_centered_crop(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 거리 변환으로 중심 찾기
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 배경이 흰색이면 반전
    if thresh[0, 0] == 255: thresh = cv2.bitwise_not(thresh)
        
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    best_cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
    cx, cy = max_loc
    
    # 패딩 및 크롭
    half = CROP_SIZE // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    
    # 패딩 추가
    pad_l = abs(x1) if x1 < 0 else 0
    pad_t = abs(y1) if y1 < 0 else 0
    pad_r = (x2 - w) if x2 > w else 0
    pad_b = (y2 - h) if y2 > h else 0
    
    if any([pad_l, pad_t, pad_r, pad_b]):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        x1, y1 = x1 + pad_l, y1 + pad_t
        x2, y2 = x2 + pad_l, y2 + pad_t
        
    return img[y1:y2, x1:x2]

# ==========================================
# 4. [핵심 로직] 도넛 영역 마스킹 & 톱니 산 추출
# ==========================================
def extract_inner_spline_peaks(img, filename):
    cx, cy = CROP_SIZE // 2, CROP_SIZE // 2
    
    # 1. HSV + CLAHE (명암비 극대화)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    
    # 2. 도넛 마스크 생성 (겉 테두리 제거용)
    # 이 마스크가 '겉 테두리'를 가려주므로 알고리즘은 안쪽 톱니만 보게 됨
    donut_mask = np.zeros_like(v_clahe)
    cv2.circle(donut_mask, (cx, cy), OUTER_LIMIT_RADIUS, 255, -1) # 바깥 한계
    cv2.circle(donut_mask, (cx, cy), INNER_LIMIT_RADIUS, 0, -1)   # 안쪽 한계
    
    # 3. 마스크 적용된 이미지 생성
    masked_v = cv2.bitwise_and(v_clahe, v_clahe, mask=donut_mask)
    
    # 4. 이진화 (Adaptive Threshold)
    # 톱니 산(밝음/금속)과 틈(어두움) 분리
    # 배경이 검은색(0)이 되었으므로, 금속 부분(톱니)을 흰색으로 잡아야 함 -> THRESH_BINARY
    _, binary = cv2.threshold(masked_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  ❌ 톱니 컨투어 검출 실패: {filename}")
        return img, 0
        
    # [중요] 도넛 영역 안에서 가장 큰 덩어리를 찾음 -> 이게 바로 톱니바퀴 링!
    # 겉 테두리는 마스크로 지워졌으므로 잡힐 수 없음.
    spline_contour = max(contours, key=cv2.contourArea)
    
    # 6. Convex Hull로 산(Peak) 찾기
    hull = cv2.convexHull(spline_contour, returnPoints=True)
    hull_points = hull.squeeze()
    
    # 피크 필터링 (너무 가까운 점 제거)
    final_peaks = []
    if len(hull_points) > 0:
        # 거리순 정렬이 아니므로 순차적으로 돌면서 거리가 먼 것만 남김
        # (간단한 로직: 점들을 순회하며 이전 점과 거리가 멀면 추가)
        # 더 정교하게 하려면 각도나 중심 거리 등을 따져야 하지만 일단 거리로 필터링
        
        # Hull 포인트들은 시계/반시계 방향으로 정렬되어 있음
        final_peaks.append(hull_points[0])
        for i in range(1, len(hull_points)):
            pt = hull_points[i]
            if np.linalg.norm(pt - final_peaks[-1]) > MIN_PEAK_DISTANCE:
                final_peaks.append(pt)
            
            # 마지막 점과 첫 점 비교 (원형이므로)
            if len(final_peaks) > 1 and np.linalg.norm(final_peaks[-1] - final_peaks[0]) < MIN_PEAK_DISTANCE:
                final_peaks.pop()

    # 7. 시각화
    result_img = img.copy()
    
    # (A) 검색 범위 표시 (파란색 원) - 이 안에서만 찾았다는 증거
    cv2.circle(result_img, (cx, cy), OUTER_LIMIT_RADIUS, (255, 0, 0), 2)
    cv2.circle(result_img, (cx, cy), INNER_LIMIT_RADIUS, (255, 0, 0), 2)
    
    # (B) 찾은 톱니 링 외곽선 (보라색)
    cv2.drawContours(result_img, [spline_contour], -1, (255, 0, 255), 2)
    
    # (C) 톱니 산 꼭지점 (빨간점)
    for pt in final_peaks:
        cv2.circle(result_img, tuple(pt), 6, (0, 0, 255), -1)
        
    count = len(final_peaks)
    cv2.putText(result_img, f"Peaks: {count}", (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    print(f"  - {filename} -> 톱니 산 검출: {count}개")
    return result_img, count

# ==========================================
# 5. 실행 로직
# ==========================================
def run_process(root_folder):
    print("🚀 [V12] 겉 테두리 제거 및 톱니 산 정밀 추출 시작")
    
    for root, dirs, files in os.walk(root_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue

        try:
            relative_path = os.path.relpath(root, input_root_folder)
        except:
            relative_path = os.path.basename(root)
            
        save_path = os.path.join(output_root_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n📂 처리 중: {relative_path}")

        for file in bmp_files:
            file_path = os.path.join(root, file)
            img = imread_korean(file_path)
            if img is None: continue
            
            # 1. 센터 크롭
            cropped = get_centered_crop(img)
            if cropped is None: continue
            
            # 2. 톱니 추출
            result_img, count = extract_inner_spline_peaks(cropped, file)
            
            # 3. 저장
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)