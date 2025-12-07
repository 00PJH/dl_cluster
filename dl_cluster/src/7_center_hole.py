import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\center_hole'

# 크롭할 크기
CROP_SIZE = 1300

# [핵심] 센터홀(중앙 구멍)의 예상 반지름 범위 (픽셀)
# 너무 작은 점(노이즈)이나 너무 큰 원(톱니 테두리)을 제외하기 위함
# 대략 50 ~ 150 사이일 확률이 높음 (이미지 해상도에 따라 조절)
HOLE_MIN_RADIUS = 50
HOLE_MAX_RADIUS = 200
# 화이트닝 범위 (안쪽 지우기, 바깥쪽 지우기)
INNER_MASK_RADIUS = 180 
OUTER_MASK_RADIUS = 650

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
# 3. Step 1: "중앙에서 가장 가까운 작은 원" 찾기
# ==========================================
def find_center_hole(img, filename):
    h, w = img.shape[:2]
    img_cx, img_cy = w // 2, h // 2
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 전처리: 센터홀을 잘 잡기 위해 블러 + 이진화
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 센터홀은 보통 어두움 -> Adaptive Threshold로 어두운 영역 추출
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # 2. 컨투어 탐색
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    best_center = (w // 2, h // 2) # 못 찾으면 그냥 이미지 중앙
    min_dist_from_img_center = float('inf')
    found = False

    for cnt in contours:
        # (A) 최소 외접원 구하기
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)
        
        # (B) 크기 필터: 우리가 찾는 '작은 원(구멍)'의 크기인가?
        if not (HOLE_MIN_RADIUS < radius < HOLE_MAX_RADIUS):
            continue
            
        # (C) 모양 필터: 진짜 원에 가까운가? (찌그러진 그림자 제외)
        area = cv2.contourArea(cnt)
        circle_area = np.pi * (radius ** 2)
        if circle_area == 0: continue
        circularity = area / circle_area # 1에 가까울수록 원
        
        if circularity < 0.6: # 0.6 미만이면 찌그러진 것
            continue
            
        # (D) 위치 필터: 이미지 중앙에서 얼마나 가까운가?
        # 사용자 아이디어: "사진의 중앙에서 시작해서..."
        dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        
        # 이미지 중앙에서 너무 먼(400px 이상) 구멍은 무시
        if dist > 400: 
            continue

        # 가장 중앙에 가까운 후보 업데이트
        if dist < min_dist_from_img_center:
            min_dist_from_img_center = dist
            best_center = (cx, cy)
            found = True
            # print(f"  - 후보 발견: R={radius}, 거리={dist:.1f}")

    if found:
        print(f"  - {filename} -> 센터홀 기준 중심 잡기 성공: {best_center}")
    else:
        print(f"  ❌ {filename} -> 센터홀 검출 실패 (이미지 중앙 사용)")
        
    return best_center

# ==========================================
# 4. Step 2: 패딩 크롭 (이미지 잘림 방지)
# ==========================================
def pad_and_crop(img, cx, cy, size):
    h, w = img.shape[:2]
    half = size // 2
    
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    
    pad_top = abs(y1) if y1 < 0 else 0
    pad_bottom = (y2 - h) if y2 > h else 0
    pad_left = abs(x1) if x1 < 0 else 0
    pad_right = (x2 - w) if x2 > w else 0
    
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
        
    cropped = img[y1:y2, x1:x2]
    if cropped.shape[:2] != (size, size):
        cropped = cv2.resize(cropped, (size, size))
    return cropped

# ==========================================
# 5. Step 3: 화이트 마스킹
# ==========================================
def process_white_masking(img, filename):
    cx, cy = CROP_SIZE // 2, CROP_SIZE // 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 톱니 추출 (적응형 이진화)
    binary = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 5)
    
    # 마스킹 (내부/외부 지우기)
    mask_outer = np.full_like(binary, 255)
    cv2.circle(mask_outer, (cx, cy), OUTER_MASK_RADIUS, 0, -1) # 외부 제한
    
    # 내부 지우기 (흰색 덧칠)
    cv2.circle(binary, (cx, cy), INNER_MASK_RADIUS, 255, -1)
    
    # 외부 지우기 (흰색 덧칠)
    final_view = cv2.bitwise_or(binary, mask_outer)
    
    # 톱니 개수 확인 및 시각화
    inverted = cv2.bitwise_not(final_view)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = cv2.cvtColor(final_view, cv2.COLOR_GRAY2BGR)
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            count += 1
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                tcx, tcy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.circle(result_img, (tcx, tcy), 4, (0, 0, 255), -1)

    # 파란색 범위 표시
    cv2.circle(result_img, (cx, cy), INNER_MASK_RADIUS, (255, 0, 0), 2)
    cv2.circle(result_img, (cx, cy), OUTER_MASK_RADIUS, (255, 0, 0), 2)
    
    cv2.putText(result_img, f"Count: {count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    return result_img

# ==========================================
# 6. 실행
# ==========================================
def run_process(root_folder):
    print("🚀 [V9] 센터홀 기준 중심 잡기 시작")
    
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
            
            # 1. [핵심] 중앙의 작은 원(센터홀) 찾기
            cx, cy = find_center_hole(img, file)
            
            # 2. 크롭
            cropped_img = pad_and_crop(img, cx, cy, CROP_SIZE)
            
            # 3. 마스킹
            result_img = process_white_masking(cropped_img, file)
            
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)