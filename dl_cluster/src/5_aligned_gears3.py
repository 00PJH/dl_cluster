import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수 (사용자 환경에 맞춰 조절)
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\aligned_gears_v3'

# [핵심 설정] 톱니바퀴의 '깊이' (픽셀 단위)
# 바깥쪽 원에서 안쪽으로 얼마만큼 들어와서 자를 것인가?
# 이 값이 너무 크면 톱니까지 지워지고, 너무 작으면 중앙 노이즈가 남음.
# 사진을 보며 40~80 사이에서 조절 필요.
TOOTH_DEPTH = 60 

# 최종 결과 이미지 크기
FINAL_SIZE = 1300

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
# 3. Step 1: 가장 바깥쪽 원(Outer Ring) 찾기
# ==========================================
def detect_outer_ring_and_crop(img, filename):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거 (가우시안 블러)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # [핵심] 가장 큰 원 찾기 (HoughCircles)
    # param2를 조절하여 원 검출 민감도 설정 (높으면 완벽한 원만 찾음)
    # minRadius를 크게 주어 작은 구멍들은 아예 무시함.
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w/2,
                               param1=100, param2=40, minRadius=int(w/4), maxRadius=int(w/1.8))
    
    cx, cy, r = w // 2, h // 2, w // 3 # 기본값 (실패 시)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 가장 큰 원 하나만 선택 (이게 바깥 테두리일 확률 99%)
        largest_circle = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[0]
        cx, cy, r = int(largest_circle[0]), int(largest_circle[1]), int(largest_circle[2])
        # print(f"  - {filename} 바깥 원 검출: ({cx}, {cy}), R={r}")
    else:
        print(f"  ❌ {filename} 바깥 원 검출 실패 (중앙값 사용)")

    # --- 크롭 및 센터링 ---
    # 찾은 원을 중심으로 캔버스 중앙으로 이동
    new_canvas = np.zeros((FINAL_SIZE, FINAL_SIZE, 3), dtype=np.uint8)
    half_size = FINAL_SIZE // 2
    
    src_x1 = cx - half_size
    src_y1 = cy - half_size
    src_x2 = cx + half_size
    src_y2 = cy + half_size
    
    dst_x1, dst_y1 = 0, 0
    dst_x2, dst_y2 = FINAL_SIZE, FINAL_SIZE
    
    # 경계 처리
    if src_x1 < 0: dst_x1, src_x1 = -src_x1, 0
    if src_y1 < 0: dst_y1, src_y1 = -src_y1, 0
    if src_x2 > w: dst_x2, src_x2 = FINAL_SIZE - (src_x2 - w), w
    if src_y2 > h: dst_y2, src_y2 = FINAL_SIZE - (src_y2 - h), h

    if (src_x2 > src_x1) and (src_y2 > src_y1):
        new_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        return new_canvas, r # 찾은 반지름도 함께 반환
    return None, 0

# ==========================================
# 4. Step 2: 도넛 마스킹 (내부 지우기) 및 분석
# ==========================================
def process_donut_and_count(img, outer_radius, filename):
    # 이미지는 이미 중앙(300, 300)으로 정렬됨
    cx, cy = FINAL_SIZE // 2, FINAL_SIZE // 2
    
    # outer_radius가 너무 크거나 작으면 보정 (크롭 과정에서 스케일은 유지됨)
    # 만약 원본에서의 r이 너무 컸다면 여기서도 클 수 있으므로 안전장치
    if outer_radius == 0: outer_radius = 250 # 기본값
    
    # 안쪽 원의 반지름 계산 (바깥 원 - 톱니 깊이)
    inner_radius = outer_radius - TOOTH_DEPTH
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- [핵심] 도넛 마스크 생성 ---
    # 사용자 요청: "톱니바퀴 산 안쪽 면을 흰색으로 블러처리"
    # 여기서는 확실한 분리를 위해 안쪽을 255(흰색)으로 채웁니다.
    # 배경이 검은색이므로, 안쪽을 흰색으로 채우면 톱니만 '검은 띠' 혹은 '중간색'으로 남습니다.
    # 하지만 톱니 추출을 위해서는 '검은 배경에 톱니만 남기는 것'이 유리하므로,
    # 여기서는 "관심 영역 밖을 지우는 방식"을 사용합니다.
    
    mask = np.zeros_like(gray)
    
    # 1. 바깥 원 그리기 (여기까지가 제품)
    cv2.circle(mask, (cx, cy), outer_radius, 255, -1)
    
    # 2. 안쪽 원 그리기 (여기는 중앙 빈 공간 + 노이즈) -> 0으로 지움
    cv2.circle(mask, (cx, cy), inner_radius, 0, -1)
    
    # 3. 마스크 적용: 도넛 영역만 남기고 나머지는 검은색(0) 처리
    # (흰색으로 채우고 싶으면 cv2.bitwise_or 등을 응용 가능하나 검출엔 Black이 유리)
    donut_img = cv2.bitwise_and(gray, gray, mask=mask)
    
    # --- 톱니 검출 ---
    # 톱니가 선명하게 남았으므로 이진화 수행
    # 톱니 부분이 밝은 금속이라면 THRESH_BINARY, 어두운 틈이라면 INV
    # 적응형 이진화가 가장 강건함
    binary = cv2.adaptiveThreshold(donut_img, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 5)
    
    # 마스크 다시 적용 (이진화 노이즈 제거)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    result_img = img.copy()
    
    # 시각화: 도넛 범위 표시 (파란색)
    cv2.circle(result_img, (cx, cy), outer_radius, (255, 0, 0), 2) # 바깥 기준선
    cv2.circle(result_img, (cx, cy), inner_radius, (255, 0, 0), 2) # 안쪽 커팅선
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30: continue # 노이즈 제거
        
        # 톱니 중심점
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        tcx = int(M["m10"] / M["m00"])
        tcy = int(M["m01"] / M["m00"])
        
        # 톱니가 도넛 띠 안에 있는지 확인
        dist = np.sqrt((tcx - cx)**2 + (tcy - cy)**2)
        if inner_radius - 10 <= dist <= outer_radius + 10:
            valid_contours.append(cnt)
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 1) # 초록색 윤곽
            cv2.circle(result_img, (tcx, tcy), 2, (0, 0, 255), -1)  # 빨간점

    count = len(valid_contours)
    
    # 텍스트
    cv2.putText(result_img, f"Count: {count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    print(f"  - {filename} -> R_out:{outer_radius}, R_in:{inner_radius}, 개수:{count}")
    return result_img

# ==========================================
# 5. 실행 로직
# ==========================================
def run_process(root_folder):
    print("🚀 [V5] 외곽 링 기준 도넛 마스킹 시작")
    
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
            
            # 1. 바깥 원 찾고 크롭 (반지름 r도 받아옴)
            cropped_img, r = detect_outer_ring_and_crop(img, file)
            
            if cropped_img is None:
                continue
            
            # 2. 도넛 마스킹 및 분석
            result_img = process_donut_and_count(cropped_img, r, file)
            
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)