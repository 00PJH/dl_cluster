import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수 (사용자 환경에 맞춰 조절)
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\aligned_gears_v2'

# [설정] 최종 결과 이미지 크기 (정사각형)
FINAL_SIZE = 1500

# [설정] 톱니 검출을 위한 도넛 영역 반지름 (중심 정렬 후 기준)
# 이미지를 600x600으로 만들고 중앙(300,300)에 기어를 놓았을 때의 거리입니다.
# 결과 이미지를 보고 파란 원이 톱니를 잘 감싸도록 조절하세요.
RADIUS_INNER = 480  # 이 안쪽은 블러 처리됨 (톱니 시작점보다 살짝 안쪽)
RADIUS_OUTER = 620  # 이 바깥쪽은 무시됨 (톱니 끝점보다 살짝 바깥쪽)

# 톱니로 인식할 최소 면적 (노이즈 제거용)
MIN_TOOTH_AREA = 150

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
# 3. 핵심 기능: 기어 '자체'의 기하학적 중심 찾기
# ==========================================
def align_gear_geometric_center(img, filename):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 강력한 전처리 (노이즈/그림자 제거)
    # 가우시안 블러로 자잘한 노이즈를 뭉갭니다.
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 2. 적응형 이진화 (Adaptive Threshold) - 중요!
    # 조명이 불균일해도 물체의 형태를 잘 잡아냅니다.
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 5)
    
    # 3. 외곽선(Contour) 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  ❌ 컨투어 없음: {filename}")
        return None

    # 가장 큰 덩어리가 기어라고 가정
    gear_contour = max(contours, key=cv2.contourArea)
    
    # [핵심 변경] 무게 중심(Moments) 대신 '최소 외접원'의 중심 사용
    # 기어 테두리를 감싸는 가장 작은 원을 찾으면, 그 원의 중심이 기어의 기하학적 중심입니다.
    # 이 방식은 내부 구멍이나 그림자에 영향을 받지 않습니다.
    (x, y), radius = cv2.minEnclosingCircle(gear_contour)
    cx, cy = int(x), int(y)
    
    # 디버깅용: 원본 이미지상에서 찾은 중심 출력
    # print(f"  - {filename} 원본 중심: ({cx}, {cy})")
    
    # 4. 이미지 중앙으로 이동 (Centering)
    new_canvas = np.zeros((FINAL_SIZE, FINAL_SIZE, 3), dtype=np.uint8)
    
    # 원본에서 가져올 범위 계산
    half_size = FINAL_SIZE // 2
    
    src_x1 = cx - half_size
    src_y1 = cy - half_size
    src_x2 = cx + half_size
    src_y2 = cy + half_size
    
    # 캔버스 복사 위치 초기화
    dst_x1, dst_y1 = 0, 0
    dst_x2, dst_y2 = FINAL_SIZE, FINAL_SIZE
    
    # 경계 검사 (이미지 밖으로 나가는 경우 처리)
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
        
    # 유효한 영역만 복사
    if (src_x2 > src_x1) and (src_y2 > src_y1):
        new_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        return new_canvas
    else:
        return None

# ==========================================
# 4. 핵심 기능: 블러 처리 및 톱니 분석
# ==========================================
def analyze_gear_features(img, filename):
    # 정렬된 이미지의 중심은 항상 (300, 300)
    cx, cy = FINAL_SIZE // 2, FINAL_SIZE // 2
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- Step A: 중앙부 블러 처리 (노이즈 제거) ---
    mask_inner = np.zeros_like(gray)
    # 안쪽 원(몸통) 영역 마스크
    cv2.circle(mask_inner, (cx, cy), RADIUS_INNER, 255, -1)
    
    # 전체 블러 이미지
    blurred_img = cv2.GaussianBlur(gray, (25, 25), 0)
    
    # 합성: 안쪽은 블러, 바깥쪽(톱니)은 원본
    processed_gray = np.where(mask_inner > 0, blurred_img, gray)
    
    # --- Step B: 톱니바퀴 산 검출 ---
    # 블러 처리된 이미지로 이진화 수행
    # ADAPTIVE_THRESH_MEAN_C 또는 GAUSSIAN_C 사용
    # 톱니 부분(어두운 틈 vs 밝은 산)을 명확히 분리
    binary = cv2.adaptiveThreshold(processed_gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)
    
    # 도넛 모양 ROI 마스크 생성
    roi_mask = np.zeros_like(gray)
    cv2.circle(roi_mask, (cx, cy), RADIUS_OUTER, 255, -1)
    cv2.circle(roi_mask, (cx, cy), RADIUS_INNER, 0, -1)
    
    # ROI 적용
    binary_roi = cv2.bitwise_and(binary, binary, mask=roi_mask)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    
    result_img = img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_TOOTH_AREA: continue
        
        # 톱니의 중심점 구하기
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        tcx = int(M["m10"] / M["m00"])
        tcy = int(M["m01"] / M["m00"])
        
        # 거리 체크 (한 번 더 확인)
        dist = np.sqrt((tcx - cx)**2 + (tcy - cy)**2)
        if RADIUS_INNER - 10 <= dist <= RADIUS_OUTER + 10:
            valid_contours.append(cnt)
            
            # 톱니 윤곽선 그리기 (초록색)
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            # 톱니 중심점 찍기 (빨간색)
            cv2.circle(result_img, (tcx, tcy), 3, (0, 0, 255), -1)

    # --- Step C: 시각화 (요청사항 반영) ---
    # [수정] 범위 표시 원을 파란색(Blue)으로 변경 (BGR: 255, 0, 0)
    # 안쪽 원 (블러 경계)
    cv2.circle(result_img, (cx, cy), RADIUS_INNER, (255, 0, 0), 2)
    # 바깥쪽 원 (검사 한계)
    cv2.circle(result_img, (cx, cy), RADIUS_OUTER, (255, 0, 0), 2)
    
    # 결과 텍스트
    count = len(valid_contours)
    cv2.putText(result_img, f"Teeth: {count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    print(f"  - {filename} -> 톱니 개수: {count}")
    return result_img

# ==========================================
# 5. 실행 로직
# ==========================================
def run_process(root_folder):
    print("🚀 [V4] 기하학적 중심 보정 + 파란색 ROI + 톱니 분석 시작")
    
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
            
            # 1. 기하학적 중심 정렬
            aligned_img = align_gear_geometric_center(img, file)
            
            if aligned_img is None:
                print(f"  ❌ 기어 검출 실패: {file}")
                continue
            
            # 2. 분석
            result_img = analyze_gear_features(aligned_img, file)
            
            # 3. 저장
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)