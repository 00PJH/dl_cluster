import cv2
import numpy as np
import os
import math

# ==========================================
# 1. 설정 변수 (이 숫자가 제일 중요합니다!)
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\use_pjh_method'

# 톱니바퀴가 위치한 '거리 범위' (중심점 기준 픽셀 거리)
# 파란색 선으로 표시될 영역입니다. 이 안에 있는 물체만 잡습니다.
search_radius_min = 150  # 톱니가 시작되는 안쪽 거리
search_radius_max = 600  # 톱니가 끝나는 바깥쪽 거리

# 톱니 하나의 대략적인 면적 (노이즈 제거용)
min_tooth_area = 50    # 너무 작은 점 무시
max_tooth_area = 2000  # 너무 큰 덩어리 무시

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
# 3. 핵심 로직: 범위 기반 톱니 추출
# ==========================================
def detect_inner_teeth(img, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 이미지 중심 찾기 (전체 제품의 센터)
    _, thresh_center = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours_center, _ = cv2.findContours(thresh_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_center:
        return img, 0

    # 가장 큰 덩어리(제품 전체)의 중심을 구함
    main_body = max(contours_center, key=cv2.contourArea)
    M = cv2.moments(main_body)
    if M["m00"] == 0: return img, 0
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # 2. 톱니 검출을 위한 이진화
    # 톱니는 보통 배경보다 어둡거나(구멍) 밝습니다(금속).
    # 여기서는 '기와집 모양'인 금속 부분을 잡기 위해 밝은 곳을 찾거나, 
    # 역으로 어두운 틈새를 찾아 반전시킬 수 있습니다.
    # (일단 Otsu로 밝은 금속 부위 분리 시도)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. 모든 내부 컨투어 찾기 (RETR_LIST 사용)
    # TREE나 LIST를 써야 안쪽에 있는 톱니들을 찾을 수 있음
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_teeth = []
    
    for cnt in contours:
        # 각 컨투어의 면적 계산
        area = cv2.contourArea(cnt)
        if not (min_tooth_area < area < max_tooth_area):
            continue # 면적이 너무 작거나 크면 패스

        # 각 컨투어의 '중심 위치' 계산
        M_t = cv2.moments(cnt)
        if M_t["m00"] == 0: continue
        t_cx = int(M_t["m10"] / M_t["m00"])
        t_cy = int(M_t["m01"] / M_t["m00"])
        
        # [핵심] 제품 중심(cx, cy)에서 이 컨투어(t_cx, t_cy)까지의 거리 계산
        dist = math.sqrt((t_cx - cx)**2 + (t_cy - cy)**2)
        
        # 거리가 우리가 설정한 '톱니 라인(search_radius)' 안에 있는지 확인
        if search_radius_min < dist < search_radius_max:
            detected_teeth.append(cnt)

    # 4. 시각화
    result_img = img.copy()
    
    # (A) 검색 범위(Search Zone) 그리기 (파란색 원 2개 - 디버깅용)
    # 이 두 원 사이에 있는 것만 잡습니다.
    cv2.circle(result_img, (cx, cy), search_radius_min, (255, 0, 0), 1)
    cv2.circle(result_img, (cx, cy), search_radius_max, (255, 0, 0), 1)
    
    # (B) 찾은 톱니들 그리기 (초록색 채우기)
    cv2.drawContours(result_img, detected_teeth, -1, (0, 255, 0), 2)
    
    # (C) 중심점
    cv2.circle(result_img, (cx, cy), 5, (0, 0, 255), -1)

    count = len(detected_teeth)
    
    # 텍스트 표시
    cv2.putText(result_img, f"Teeth Count: {count}", (cx - 80, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    print(f"  - {filename} -> 검출된 톱니 수: {count}")
    return result_img, count

# ==========================================
# 4. 실행 로직
# ==========================================
def run_process(root_folder):
    print(f"🚀 [Inner Teeth Detection] 분석 시작...")
    
    for root, dirs, files in os.walk(root_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue

        folder_name = os.path.basename(root)
        try:
            relative_path = os.path.relpath(root, input_root_folder)
        except:
            relative_path = folder_name
            
        # use_pjh_method 폴더에 저장
        save_path = os.path.join(output_root_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n📂 처리 중인 폴더: {relative_path}")

        for file in bmp_files:
            file_path = os.path.join(root, file)
            img = imread_korean(file_path)
            if img is None: continue
            
            result_img, count = detect_inner_teeth(img, file)
            
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)