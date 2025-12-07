import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수 (사용자 조정)
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\white_teeth_v8'

# 크롭할 이미지 크기 (넉넉하게 설정)
CROP_SIZE = 1300

# [핵심] 흰색으로 덮어버릴 내부 원의 반지름
# 이 값보다 안쪽은 무조건 흰색이 되어 노이즈가 사라짐.
# 결과 이미지를 보고 톱니가 시작되기 직전까지 이 값을 키우면 됨.
INNER_MASK_RADIUS = 180 

# 바깥쪽 제한 반지름 (이 밖도 흰색 처리)
OUTER_MASK_RADIUS = 600

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
# 3. Step 1: 기어 전체 덩어리 중심 찾기
# ==========================================
def find_gear_center(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거 및 이진화
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    # 배경과 기어를 분리 (어두운 기어 찾기)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return w//2, h//2

    # 가장 큰 덩어리(기어) 선택
    best_cnt = max(contours, key=cv2.contourArea)
    
    # [중요] 최소 외접원 중심 사용 (그림자에 강함)
    (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
    
    return int(cx), int(cy)

# ==========================================
# 4. Step 2: 안전한 패딩 크롭 (Safe Crop)
# ==========================================
def pad_and_crop(img, cx, cy, size):
    h, w = img.shape[:2]
    half = size // 2
    
    # 1. 크롭할 좌표 계산
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half
    
    # 2. 패딩이 필요한지 계산
    pad_top = abs(y1) if y1 < 0 else 0
    pad_bottom = (y2 - h) if y2 > h else 0
    pad_left = abs(x1) if x1 < 0 else 0
    pad_right = (x2 - w) if x2 > w else 0
    
    # 3. 이미지에 패딩 추가 (흰색 배경으로 확장)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # 좌표 보정 (패딩만큼 중심 이동)
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
        
    # 4. 안전하게 자르기
    cropped = img[y1:y2, x1:x2]
    
    # 만약 크기가 안 맞으면 강제 리사이즈 (안전장치)
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size))
        
    return cropped

# ==========================================
# 5. Step 3: 화이트 마스킹 및 톱니 검출
# ==========================================
def process_white_masking(img, filename):
    # 이미지는 CROP_SIZE 정중앙에 기어가 위치함
    cx, cy = CROP_SIZE // 2, CROP_SIZE // 2
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 적응형 이진화 (톱니 찾기)
    # 주변보다 어두운 부분(톱니)을 찾음
    binary = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 25, 5)
    
    # 2. [핵심] 화이트 마스크 적용 (White-out)
    # 내부와 외부를 흰색(255)으로 덮어씀
    
    # (A) 내부 마스크 (Inner Circle)
    cv2.circle(binary, (cx, cy), INNER_MASK_RADIUS, 255, -1)
    
    # (B) 외부 마스크 (Outer Area)
    # 이미지 전체를 흰색으로 채운 마스크를 만들고, 도넛 영역만 뚫어서 합성
    mask_outer = np.full_like(binary, 255)
    cv2.circle(mask_outer, (cx, cy), OUTER_MASK_RADIUS, 0, -1) # 바깥 한계 안쪽을 0으로
    
    # binary 이미지와 mask_outer를 합침 (OR 연산: 하나라도 255면 255)
    final_view = cv2.bitwise_or(binary, mask_outer)
    
    # 3. 톱니 개수 세기 (검은색 덩어리 찾기)
    # 검은색 톱니를 찾기 위해 반전 후 컨투어 탐색
    inverted = cv2.bitwise_not(final_view)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    result_img = cv2.cvtColor(final_view, cv2.COLOR_GRAY2BGR)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30: # 노이즈 제거
            count += 1
            # 중심점 표시 (빨간점)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                tcx, tcy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.circle(result_img, (tcx, tcy), 4, (0, 0, 255), -1)

    # 4. 시각화 (마스킹 범위 파란 원 표시)
    cv2.circle(result_img, (cx, cy), INNER_MASK_RADIUS, (255, 0, 0), 2)
    cv2.circle(result_img, (cx, cy), OUTER_MASK_RADIUS, (255, 0, 0), 2)
    
    cv2.putText(result_img, f"Count: {count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    print(f"  - {filename} -> White-out 처리 완료, 개수: {count}")
    return result_img

# ==========================================
# 6. 실행
# ==========================================
def run_process(root_folder):
    print("🚀 [V8] 안전 크롭 + 내부 화이트닝 시작")
    
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
            
            # 1. 중심 찾기
            cx, cy = find_gear_center(img)
            
            # 2. 패딩 포함 안전 크롭
            cropped_img = pad_and_crop(img, cx, cy, CROP_SIZE)
            
            # 3. 화이트 마스킹 및 분석
            result_img = process_white_masking(cropped_img, file)
            
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)