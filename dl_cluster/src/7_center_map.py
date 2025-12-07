import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\center_v10'

# 크롭할 크기
CROP_SIZE = 1400

# 화이트닝 범위 (이미지 크롭 후 중앙 기준)
INNER_MASK_RADIUS = 300 
OUTER_MASK_RADIUS = 620

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
# 3. Step 1: 거리 변환(Distance Transform)으로 중심 찾기
# ==========================================
def find_center_by_distance_transform(img, filename):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 이진화 (전처리)
    # 노이즈를 없애기 위해 블러를 좀 강하게 줍니다.
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 배경과 물체를 나눕니다. (Otsu 알고리즘 자동)
    # 기어가 어두운지 밝은지에 따라 THRESH_BINARY 또는 INV가 필요할 수 있습니다.
    # 보통 기어 금속이 밝고 배경이 어두우면 THRESH_BINARY
    # 기어가 어둡고 배경이 밝으면 THRESH_BINARY_INV
    # 여기서는 일단 "큰 덩어리"를 흰색으로 만드는 것이 목표입니다.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 만약 배경이 흰색으로 잡혔다면(가장자리가 흰색이면), 반전시켜야 합니다.
    # (기어가 흰색 덩어리가 되어야 거리 변환이 가능함)
    if thresh[0, 0] == 255:
        thresh = cv2.bitwise_not(thresh)

    # 2. 가장 큰 덩어리(기어)만 남기기
    # 자잘한 노이즈나 글자 등을 제거하기 위함
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return w//2, h//2
    
    # 가장 면적이 큰 컨투어 찾기
    best_cnt = max(contours, key=cv2.contourArea)
    
    # 깨끗한 마스크 생성 (기어만 흰색, 나머지 검은색)
    clean_mask = np.zeros_like(gray)
    cv2.drawContours(clean_mask, [best_cnt], -1, 255, -1)
    
    # 3. [핵심] 거리 변환 수행
    # 흰색 영역의 내부 픽셀들이 가장자리(검은색)에서 얼마나 먼지 계산
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    
    # 4. 최대값 위치 찾기 (가장 깊은 곳 = 기하학적 중심)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    
    cx, cy = max_loc
    print(f"  - {filename} -> 거리 변환 중심: ({cx}, {cy})")
    
    return cx, cy

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
        # 패딩은 검은색(0)보다는 흰색(255)이나 중간색으로 채우는 게 시각적으로 나음
        # 분석에는 영향 없음 (어차피 마스킹하니까)
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
        
    cropped = img[y1:y2, x1:x2]
    
    # 혹시 모를 사이즈 불일치 보정
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
    # 블러를 살짝 줘서 노이즈 줄임
    blurred_for_tooth = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred_for_tooth, 255, 
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
    # 톱니는 흰 배경(255) 위의 검은색(0)이므로 반전해서 카운트
    inverted = cv2.bitwise_not(final_view)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = cv2.cvtColor(final_view, cv2.COLOR_GRAY2BGR)
    count = 0
    for cnt in contours:
        # 너무 작은 점은 노이즈
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
    print("🚀 [V10] 거리 변환(Distance Transform) 기반 중심 잡기 시작")
    
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
            
            # 1. 거리 변환으로 중심 찾기 (가장 강력한 방법)
            cx, cy = find_center_by_distance_transform(img, file)
            
            # 2. 크롭
            cropped_img = pad_and_crop(img, cx, cy, CROP_SIZE)
            
            # 3. 마스킹
            result_img = process_white_masking(cropped_img, file)
            
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)