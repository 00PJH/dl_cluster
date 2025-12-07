import cv2
import numpy as np
import os
import math

# ==========================================
# 1. 설정 변수 (환경에 맞춰 조절하세요)
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\aligned_gears'

# 결과 이미지 크기 (정사각형으로 통일)
FINAL_SIZE = 1500

# 톱니 검출을 위한 ROI 반지름 설정 (중심 정렬 후의 픽셀 거리)
# 기어가 항상 중앙에 오므로 이 값들은 고정해도 잘 맞습니다.
# 결과 이미지를 보며 미세조정하세요.
RADIUS_INNER = 350  # 이보다 안쪽은 블러처리 & 무시
RADIUS_OUTER = 600  # 이보다 바깥쪽은 무시

# 톱니로 인식할 최소 면적
MIN_TOOTH_AREA = 30

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
# 3. 핵심 기능: 자동 중심 정렬 (Centering)
# ==========================================
def align_gear_center(img):
    """
    이미지에서 기어의 중심을 찾아 캔버스 정중앙으로 이동시킵니다.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 기어 몸체 찾기 (이진화)
    # 배경은 밝고 기어는 어둡거나, 반대일 수 있으므로 Otsu 사용
    # 만약 잘 안 잡히면 threshold 값을 직접 지정 (예: 100)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. 가장 큰 컨투어(기어 전체) 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None # 기어를 못 찾음

    gear_contour = max(contours, key=cv2.contourArea)
    
    # 3. 무게 중심 계산
    M = cv2.moments(gear_contour)
    if M["m00"] == 0: return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # 4. 새 캔버스에 중앙 배치
    # FINAL_SIZE 크기의 검은 캔버스 생성
    new_canvas = np.zeros((FINAL_SIZE, FINAL_SIZE, 3), dtype=np.uint8)
    
    # 복사할 원본 영역 계산
    half_size = FINAL_SIZE // 2
    
    # 원본에서 가져올 범위 (cx, cy 기준)
    src_x1 = cx - half_size
    src_y1 = cy - half_size
    src_x2 = cx + half_size
    src_y2 = cy + half_size
    
    # 캔버스에 붙일 범위 (기본은 전체, 원본이 잘리면 그만큼 줄어듦)
    dst_x1, dst_y1 = 0, 0
    dst_x2, dst_y2 = FINAL_SIZE, FINAL_SIZE
    
    # 범위 보정 (이미지 밖으로 나가는 경우 처리)
    if src_x1 < 0:
        dst_x1 = -src_x1
        src_x1 = 0
    if src_y1 < 0:
        dst_y1 = -src_y1
        src_y1 = 0
    if src_x2 > w:
        dst_x2 = FINAL_SIZE - (src_x2 - w)
        src_x2 = w
    if src_y2 > h:
        dst_y2 = FINAL_SIZE - (src_y2 - h)
        src_y2 = h
        
    # 이미지 복사
    if src_w := (src_x2 - src_x1) > 0 and (src_h := (src_y2 - src_y1)) > 0:
        new_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        return new_canvas
    else:
        return None

# ==========================================
# 4. 핵심 기능: 블러 처리 및 톱니 검출
# ==========================================
def analyze_centered_gear(img, filename):
    # 1. 이미 정렬된 이미지이므로 중심은 무조건 (300, 300)
    cx, cy = FINAL_SIZE // 2, FINAL_SIZE // 2
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- [Step A] 영역별 블러 처리 (노이즈 제거) ---
    # 마스크 생성: 안쪽 원(몸통)은 흰색, 바깥쪽(톱니)은 검은색
    mask_inner = np.zeros_like(gray)
    cv2.circle(mask_inner, (cx, cy), RADIUS_INNER, 255, -1)
    
    # 원본을 강하게 블러처리 (노이즈 제거용)
    blurred_img = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # 합성: 안쪽은 블러 이미지, 바깥쪽(톱니)은 선명한 원본 사용
    # 이렇게 하면 중앙의 글씨나 빛 반사가 뭉개져서 톱니로 오인받지 않음
    processed_gray = np.where(mask_inner > 0, blurred_img, gray)
    
    # --- [Step B] 톱니 검출 (ROI 방식 + Contour) ---
    # 전처리된 이미지로 이진화
    # 톱니(어두운 틈 or 밝은 산)를 잡기 위해 적응형 threshold 사용 권장
    # 상황에 따라 cv2.THRESH_BINARY_INV 등으로 변경 필요
    binary = cv2.adaptiveThreshold(processed_gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # ROI 마스크 적용 (도넛 모양)
    # 1번 코드의 장점: 톱니가 있을법한 도넛 영역만 남김
    roi_mask = np.zeros_like(gray)
    cv2.circle(roi_mask, (cx, cy), RADIUS_OUTER, 255, -1) # 바깥 원
    cv2.circle(roi_mask, (cx, cy), RADIUS_INNER, 0, -1)   # 안쪽 원 빼기
    
    # ROI 밖은 다 지워버림
    binary_roi = cv2.bitwise_and(binary, binary, mask=roi_mask)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_peaks = []
    
    # 결과 시각화용 이미지
    result_img = img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_TOOTH_AREA: continue
        
        # 각 톱니의 중심점(Peak Candidate) 구하기
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        
        tcx = int(M["m10"] / M["m00"])
        tcy = int(M["m01"] / M["m00"])
        
        # 중심에서의 거리 확인 (ROI 안에 확실히 있는지 이중 체크)
        dist = np.sqrt((tcx - cx)**2 + (tcy - cy)**2)
        if RADIUS_INNER <= dist <= RADIUS_OUTER:
            valid_peaks.append((tcx, tcy))
            
            # 2번 코드의 장점: 톱니 모양에 딱 맞게 그리기
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2) # 초록색 외곽선
            cv2.circle(result_img, (tcx, tcy), 3, (0, 0, 255), -1)  # 빨간점

    # --- [Step C] 시각화 및 정보 표시 ---
    # ROI 범위 표시 (노란색 선) - 디버깅용
    cv2.circle(result_img, (cx, cy), RADIUS_INNER, (0, 255, 255), 1)
    cv2.circle(result_img, (cx, cy), RADIUS_OUTER, (0, 255, 255), 1)
    
    count = len(valid_peaks)
    cv2.putText(result_img, f"Count: {count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    print(f"  - {filename} -> 검출된 톱니: {count}개")
    return result_img

# ==========================================
# 5. 실행 로직
# ==========================================
def run_process(root_folder):
    print("🚀 [통합 솔루션] 중심 정렬 + 블러 처리 + 톱니 분석 시작")
    
    for root, dirs, files in os.walk(root_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue

        # 폴더 구조 유지
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
            
            # 1. 중심 정렬 (Centering)
            aligned_img = align_gear_center(img)
            
            if aligned_img is None:
                print(f"  ❌ 중심 찾기 실패: {file}")
                continue
            
            # 2. 분석 실행 (Blur + ROI)
            result_img = analyze_centered_gear(aligned_img, file)
            
            # 3. 저장
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)