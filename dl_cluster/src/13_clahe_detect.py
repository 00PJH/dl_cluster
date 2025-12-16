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
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\13_clahe_separate_teeth'

# [핵심] 마스킹 비율 설정 (0.0 ~ 1.0)
# 바운딩 박스 반지름 대비 얼마나 안쪽을 파낼 것인가?
# 짙은 회색 원을 피하기 위해 0.55(55%) 이상으로 과감하게 설정합니다.
# 결과 이미지를 보고 이 값을 조절하면 됩니다.
INNER_MASK_RATIO = 0.85  
OUTER_MASK_RATIO = 1.0   # 박스 크기만큼 꽉 채워서 검사

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
# 4. 핵심 로직: CLAHE + 도넛 마스킹 + 개별 컨투어
# ==========================================
def process_separate_teeth(img, filename, save_dirs):
    # 1. YOLO 추론
    results = model.predict(img, conf=0.5, verbose=False)
    
    if len(results[0].boxes) == 0:
        print(f"❌ {filename} -> YOLO 미검출")
        return

    # 2. 중심 및 박스 크기 계산
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    # 박스의 반지름(짧은 쪽 기준) 계산 -> 기어의 대략적 크기
    box_radius = min(x2 - x1, y2 - y1) // 2

    # 3. 크롭 (중심 기준 1500x1500)
    h, w = img.shape[:2]
    half = CROP_SIZE // 2
    
    # 패딩 및 크롭 로직
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
    
    # 크롭된 이미지의 중심
    center_crop = (CROP_SIZE // 2, CROP_SIZE // 2)
    
    # [Step 0] 크롭 원본 저장
    imwrite_korean(os.path.join(save_dirs['0'], filename), cropped)

    # 4. CLAHE 적용 (어두운 톱니 강조)
    # HSV 변환 -> V채널 추출
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    
    # CLAHE 적용 (대비 극대화: 톱니는 더 어둡게, 회색 원은 더 밝게 분리 유도)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_ch)
    
    # [Step 1] CLAHE 결과 저장 (눈으로 확인용)
    imwrite_korean(os.path.join(save_dirs['1'], filename), v_clahe)

    # 5. 동적 임계값 (Otsu Thresholding)
    # 가장 어두운 영역(톱니)을 자동으로 찾음
    # THRESH_BINARY_INV: 어두운 곳을 흰색(255)으로, 밝은 곳을 검은색(0)으로
    _, binary = cv2.threshold(v_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # [Step 2] 이진화 결과 저장
    imwrite_korean(os.path.join(save_dirs['2'], filename), binary)

    # 6. 도넛 마스킹 (내부 회색 원 제거)
    # YOLO 박스 크기 기반으로 마스크 반지름 결정
    mask = np.zeros_like(binary)
    
    r_inner = int(box_radius * INNER_MASK_RATIO) # 안쪽 구멍 (회색 원 제거용)
    r_outer = int(box_radius * OUTER_MASK_RATIO) # 바깥 원 (톱니 포함)
    
    # 도넛 그리기 (흰색 영역만 남김)
    cv2.circle(mask, center_crop, r_outer, 255, -1)
    cv2.circle(mask, center_crop, r_inner, 0, -1)
    
    # 마스크 적용
    masked_binary = cv2.bitwise_and(binary, binary, mask=mask)
    
    # [Step 3] 마스킹 후 결과 저장
    imwrite_korean(os.path.join(save_dirs['3'], filename), masked_binary)

    # 7. 모폴로지 (노이즈 제거 및 덩어리 정리)
    kernel = np.ones((5, 5), np.uint8)
    processed_binary = cv2.morphologyEx(masked_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    processed_binary = cv2.morphologyEx(processed_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # [Step 4] 모폴로지 결과 저장
    imwrite_korean(os.path.join(save_dirs['4'], filename), processed_binary)

    # 8. 개별 테두리 추출
    contours, _ = cv2.findContours(processed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_img = cropped.copy()
    count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 너무 작은 노이즈(먼지) 제거
        if area > 100: 
            count += 1
            # 각각의 테두리를 보라색으로 그림
            cv2.drawContours(final_img, [cnt], -1, (255, 0, 255), 2)
            
            # 중심점 표시 (선택 사항)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx_t = int(M["m10"] / M["m00"])
                cy_t = int(M["m01"] / M["m00"])
                cv2.circle(final_img, (cx_t, cy_t), 3, (0, 0, 255), -1)

    # 마스크 범위 시각화 (파란원)
    cv2.circle(final_img, center_crop, r_inner, (255, 0, 0), 2)
    cv2.circle(final_img, center_crop, r_outer, (255, 0, 0), 2)
    
    cv2.putText(final_img, f"Teeth: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    
    # [Step 5] 최종 결과 저장
    imwrite_korean(os.path.join(save_dirs['5'], filename), final_img)
    
    print(f"✅ {filename} -> 톱니 개수: {count}")

# ==========================================
# 5. 실행 로직
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [CLAHE + Separate Teeth Extraction] 시작")
    
    # 단계별 폴더 생성
    step_folders = ['0_crop', '1_clahe_v', '2_binary_otsu', '3_masked', '4_morphology', '5_final_contours']
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
            
        # 하위 폴더 구조 생성
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
            
            process_separate_teeth(img, file, current_save_dirs)

    print("\n✅ 모든 작업이 완료되었습니다. 13_clahe_separate_teeth 폴더를 확인하세요.")