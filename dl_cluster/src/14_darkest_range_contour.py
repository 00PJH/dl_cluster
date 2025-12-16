import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# 1. 설정 변수
# ==========================================
# 모델 경로
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\train_yolo8n_sizeup_boundingbox\weights\best.pt'

# 입력 및 출력 폴더
input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\14_darkest_range_contour'

# [핵심] 거리 필터링 비율 (0.0 ~ 1.0)
# YOLO 박스 반지름 기준, 중심에서 이 비율보다 멀리 있는 덩어리만 톱니로 인정함.
# 예: 0.65라면 반지름의 65% 바깥쪽에 있는 것만 테두리를 그림. (내부 회색 원 무시)
MIN_DIST_RATIO = 0.86
MAX_DIST_RATIO = 1   # 너무 먼 노이즈 제거용

# 크롭 크기
CROP_SIZE = 1500

# ==========================================
# 2. 모델 로드
# ==========================================
print(f"🔄 모델 로딩: {model_path}")
try:
    model = YOLO(model_path)
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

# ==========================================
# 3. 유틸리티 함수
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
# 4. 핵심 로직: 어두운 영역 추출 + 거리 기반 필터링
# ==========================================
def process_dark_teeth_contour(img, filename, save_dirs):
    # 1. YOLO 추론
    results = model.predict(img, conf=0.5, verbose=False)
    
    if len(results[0].boxes) == 0:
        print(f"❌ {filename} -> YOLO 미검출")
        return

    # 2. 박스 및 중심 정보 계산
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    # 기어의 반지름 추정 (박스 짧은 변의 절반)
    gear_radius = min(x2 - x1, y2 - y1) // 2

    # 3. 크롭 (중심 기준 1500x1500)
    h, w = img.shape[:2]
    half = CROP_SIZE // 2
    
    pad_l = abs(cx - half) if (cx - half) < 0 else 0
    pad_t = abs(cy - half) if (cy - half) < 0 else 0
    pad_r = (cx + half - w) if (cx + half) > w else 0
    pad_b = (cy + half - h) if (cy + half) > h else 0
    
    if any([pad_l, pad_t, pad_r, pad_b]):
        img_padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cx += pad_l; cy += pad_t
    else:
        img_padded = img.copy()
        
    src_x1, src_y1 = cx - half, cy - half
    src_x2, src_y2 = cx + half, cy + half
    cropped = img_padded[src_y1:src_y2, src_x1:src_x2].copy()
    
    center_crop = (CROP_SIZE // 2, CROP_SIZE // 2)
    
    # [Step 0] 원본 크롭 저장
    imwrite_korean(os.path.join(save_dirs['0'], filename), cropped)

    # 4. CLAHE 적용 (어두운 부분 강조)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_ch)
    
    # [Step 1] CLAHE 결과 저장
    imwrite_korean(os.path.join(save_dirs['1'], filename), v_clahe)

    # 5. [사용자 아이디어] 가장 어두운 영역 찾기
    # Otsu 이진화 + 반전 (어두운 곳이 흰색이 됨)
    _, binary = cv2.threshold(v_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # [Step 2] 이진화 결과 저장
    imwrite_korean(os.path.join(save_dirs['2'], filename), binary)

    # 6. 모폴로지 (덩어리 분리 및 노이즈 제거)
    # 톱니와 내부 원이 살짝 붙어있을 경우를 대비해 Open 연산 수행
    kernel = np.ones((5, 5), np.uint8)
    processed_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # [Step 3] 모폴로지 결과 저장
    imwrite_korean(os.path.join(save_dirs['3'], filename), processed_binary)

    # 7. [핵심] 컨투어 추출 및 거리 기반 필터링
    contours, _ = cv2.findContours(processed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_img = cropped.copy()
    count = 0
    
    # 필터링 기준 거리 계산
    min_dist_limit = gear_radius * MIN_DIST_RATIO
    max_dist_limit = gear_radius * MAX_DIST_RATIO
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 너무 작은 노이즈 제거
        if area > 50:
            # 컨투어의 무게 중심 계산
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx_t = int(M["m10"] / M["m00"])
                cy_t = int(M["m01"] / M["m00"])
                
                # 이미지 중앙(center_crop)과의 거리 계산
                dist_from_center = np.sqrt((cx_t - center_crop[0])**2 + (cy_t - center_crop[1])**2)
                
                # [논리 적용] 거리가 '내부 원'보다 멀고 '바깥 원'보다 가까운 것만 톱니로 인정
                # 이렇게 하면 중앙에 있는 짙은 회색 원은 거리가 가까워서 탈락함
                if min_dist_limit < dist_from_center < max_dist_limit:
                    count += 1
                    # 테두리 그리기 (보라색)
                    cv2.drawContours(final_img, [cnt], -1, (255, 0, 255), 2)
                    # 중심점 표시 (빨간색)
                    cv2.circle(final_img, (cx_t, cy_t), 4, (0, 0, 255), -1)

    # [시각화] 필터링 기준선 표시 (파란색 원)
    # 이 선들 사이에 있는 어두운 덩어리만 잡았다는 의미
    cv2.circle(final_img, center_crop, int(min_dist_limit), (255, 0, 0), 2)
    cv2.circle(final_img, center_crop, int(max_dist_limit), (255, 0, 0), 2)
    
    cv2.putText(final_img, f"Teeth: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    
    # [Step 4] 최종 결과 저장
    imwrite_korean(os.path.join(save_dirs['4'], filename), final_img)
    
    print(f"✅ {filename} -> 톱니 개수: {count}")

# ==========================================
# 5. 실행 로직
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Darkest Region & Distance Filter] 분석 시작")
    
    # 단계별 폴더 생성
    step_folders = ['0_crop', '1_clahe_v', '2_binary_otsu', '3_morphology', '4_final_contours']
    save_dirs = {}
    
    for idx, folder in enumerate(step_folders):
        path = os.path.join(output_root_folder, folder)
        save_dirs[str(idx)] = path
        os.makedirs(path, exist_ok=True)
        
    for root, dirs, files in os.walk(input_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue
        
        try:
            rel_path = os.path.relpath(root, input_folder)
        except:
            rel_path = os.path.basename(root)
            
        current_save_dirs = {}
        for key, path in save_dirs.items():
            sub_path = os.path.join(path, rel_path)
            os.makedirs(sub_path, exist_ok=True)
            current_save_dirs[key] = sub_path
        
        print(f"\n📂 폴더 처리 중: {rel_path}")
        
        for file in bmp_files:
            img_path = os.path.join(root, file)
            img = imread_korean(img_path)
            if img is None: continue
            
            process_dark_teeth_contour(img, file, current_save_dirs)

    print("\n✅ 모든 작업 완료. 14_darkest_range_contour 폴더를 확인하세요.")