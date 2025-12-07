import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수 (사용자 조정 필수!)
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\aligned_gears_v4'

# [핵심 1] 톱니바퀴 바로 바깥 원의 '최대 반지름' (원본 이미지 기준)
# 배경 원(컨테이너)이 잡히지 않도록 이 값을 배경 원보다 작게 설정해야 합니다.
# 원본 해상도가 2448x2048 정도라면, 톱니바퀴 반지름은 보통 400~600 사이일 것입니다.
# 배경 원이 800이라면, 여기를 700 정도로 제한하세요.
MAX_RADIUS_LIMIT = 500 
MIN_RADIUS_LIMIT = 300  # 너무 작은 원(센터홀 등) 무시

# [핵심 2] 톱니바퀴의 '깊이' (바깥 원에서 안쪽으로 파고들 거리)
# 이 값으로 도넛의 두께를 결정합니다.
TOOTH_DEPTH = 60

# 최종 결과 이미지 크기 (요청하신 대로 크게)
FINAL_SIZE = 1500

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
# 3. Step 1: "어두운 회색 원" 정밀 검출
# ==========================================
def detect_dark_outer_ring(img, filename):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 1. 허프 변환으로 '모든' 원 후보 찾기 (param2를 낮춰서 많이 찾게 함)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=20, # 민감하게 많이 찾음
                               minRadius=MIN_RADIUS_LIMIT, 
                               maxRadius=MAX_RADIUS_LIMIT) # [중요] 배경 원 차단

    best_circle = None
    min_brightness = 255 # 어두운 원을 찾기 위한 기준
    
    debug_img = img.copy() # 디버깅용 (어떤 원들을 검사했는지 확인)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            cx, cy, r = int(i[0]), int(i[1]), int(i[2])
            
            # 필터 1: 위치 체크 (이미지 중앙 부근에 있어야 함)
            dist_from_center = np.sqrt((cx - w//2)**2 + (cy - h//2)**2)
            if dist_from_center > 200: 
                continue # 중앙에서 너무 멀면 탈락

            # 필터 2: 명암(Brightness) 체크 - "회색/검은색 원" 찾기
            # 원의 둘레(Circumference)를 따라 픽셀 값을 샘플링해서 평균 밝기를 구함
            mask_check = np.zeros_like(gray)
            cv2.circle(mask_check, (cx, cy), r, 255, 2) # 두께 2인 원 그리기
            
            # 마스크 영역의 평균 밝기 계산
            mean_val = cv2.mean(gray, mask=mask_check)[0]
            
            # 시각화 (후보들은 노란색)
            cv2.circle(debug_img, (cx, cy), r, (0, 255, 255), 1)
            
            # 가장 어두운 원을 선택 (User Requirement: 회색~검은색 원)
            # 단, 너무 어두운 것(검은 배경 자체)은 제외하려면 조건을 추가할 수 있음
            if mean_val < min_brightness:
                min_brightness = mean_val
                best_circle = (cx, cy, r)

    if best_circle:
        cx, cy, r = best_circle
        print(f"  - {filename} -> 타겟 원 검출 성공! (R={r}, 밝기={min_brightness:.1f})")
        return cx, cy, r
    else:
        print(f"  ❌ {filename} -> 적절한 원을 못 찾음 (중앙값 대체)")
        return w//2, h//2, w//4

# ==========================================
# 4. Step 2: 크롭 & 도넛 마스킹
# ==========================================
def process_gear_v6(img, cx, cy, r, filename):
    h, w = img.shape[:2]
    
    # 1. 캔버스 정중앙으로 이동 (Centering)
    new_canvas = np.zeros((FINAL_SIZE, FINAL_SIZE, 3), dtype=np.uint8)
    half_size = FINAL_SIZE // 2
    
    src_x1 = cx - half_size
    src_y1 = cy - half_size
    src_x2 = cx + half_size
    src_y2 = cy + half_size
    
    dst_x1, dst_y1 = 0, 0
    dst_x2, dst_y2 = FINAL_SIZE, FINAL_SIZE
    
    # 좌표 보정
    if src_x1 < 0: dst_x1, src_x1 = -src_x1, 0
    if src_y1 < 0: dst_y1, src_y1 = -src_y1, 0
    if src_x2 > w: dst_x2, src_x2 = FINAL_SIZE - (src_x2 - w), w
    if src_y2 > h: dst_y2, src_y2 = FINAL_SIZE - (src_y2 - h), h
    
    if (src_x2 > src_x1) and (src_y2 > src_y1):
        new_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    
    # 2. 도넛 마스킹
    # 이제 중심은 무조건 (FINAL_SIZE/2, FINAL_SIZE/2)
    center_final = (half_size, half_size)
    
    # 안쪽 원 반지름 계산 (검출된 반지름 r은 그대로 유지됨)
    inner_radius = max(10, r - TOOTH_DEPTH)
    
    gray_canvas = cv2.cvtColor(new_canvas, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_canvas)
    
    # 바깥 원 (찾은 테두리)
    cv2.circle(mask, center_final, r, 255, -1)
    # 안쪽 원 (지울 영역)
    cv2.circle(mask, center_final, inner_radius, 0, -1)
    
    # 마스크 적용 (도넛만 남김)
    donut_img = cv2.bitwise_and(gray_canvas, gray_canvas, mask=mask)
    
    # 3. 톱니 검출 및 시각화
    # 톱니 추출 (이진화)
    binary = cv2.adaptiveThreshold(donut_img, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 5)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = new_canvas.copy()
    valid_cnt = []
    
    # 시각화: 기준이 된 원 (파란색)
    cv2.circle(result_img, center_final, r, (255, 0, 0), 2)     # Outer
    cv2.circle(result_img, center_final, inner_radius, (255, 0, 0), 2) # Inner
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30: continue
        
        # 위치 필터
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        tcx = int(M["m10"] / M["m00"])
        tcy = int(M["m01"] / M["m00"])
        
        dist = np.sqrt((tcx - half_size)**2 + (tcy - half_size)**2)
        if inner_radius - 10 <= dist <= r + 10:
            valid_cnt.append(cnt)
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 1)
            
    count = len(valid_cnt)
    cv2.putText(result_img, f"Count: {count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    return result_img

# ==========================================
# 5. 실행
# ==========================================
def run_process(root_folder):
    print("🚀 [V6] '어두운 회색 원' 타겟팅 및 도넛 분석 시작")
    
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
            
            # 1. 어두운 바깥 원 검출
            cx, cy, r = detect_dark_outer_ring(img, file)
            
            # 2. 크롭 및 분석
            result_img = process_gear_v6(img, cx, cy, r, file)
            
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)