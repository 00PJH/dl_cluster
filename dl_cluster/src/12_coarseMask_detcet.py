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
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\12_coarseMask_contour'

# 내부 원(구멍)의 크기 비율 (박스 크기 대비)
INNER_RATIO = 0.8

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
# 4. 핵심 로직: 단계별 처리 및 저장
# ==========================================
def process_and_save_steps(img, filename, save_dirs):
    h, w = img.shape[:2]

    # --- [Step 0] YOLO 추론 및 시각화 ---
    results = model.predict(img, conf=0.5, verbose=False)
    
    if len(results[0].boxes) == 0:
        print(f"❌ {filename} -> YOLO 미검출")
        return

    # 좌표 계산
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    box_w, box_h = x2 - x1, y2 - y1
    
    # 0단계 이미지 저장 (박스와 예상 범위 표시)
    step0_img = img.copy()
    cv2.rectangle(step0_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # 박스
    cv2.ellipse(step0_img, (cx, cy), (box_w // 2, box_h // 2), 0, 0, 360, (255, 0, 0), 2) # 내접 타원
    inner_radius = int(min(box_w, box_h) * INNER_RATIO / 2)
    cv2.circle(step0_img, (cx, cy), inner_radius, (0, 0, 255), 2) # 내부 원
    
    imwrite_korean(os.path.join(save_dirs['0'], filename), step0_img)

    # --- [Step 1] 도넛 마스크 생성 ---
    mask_ring = np.zeros((h, w), dtype=np.uint8)
    # 타원 그리기 (흰색)
    cv2.ellipse(mask_ring, (cx, cy), (box_w // 2, box_h // 2), 0, 0, 360, 255, -1)
    # 내부 원 파내기 (검은색)
    cv2.circle(mask_ring, (cx, cy), inner_radius, 0, -1)
    
    imwrite_korean(os.path.join(save_dirs['1'], filename), mask_ring)
    
    # --- [Step 2] 화이트닝 (배경 지우기) ---
    white_bg = np.full_like(img, 255)
    # mask_ring이 있는 곳만 원본, 나머지는 흰색
    step2_whitened = np.where(mask_ring[..., None] > 0, img, white_bg)
    
    imwrite_korean(os.path.join(save_dirs['2'], filename), step2_whitened)
    
    # --- [Step 3] 이진화 (Binary Raw) ---
    gray = cv2.cvtColor(step2_whitened, cv2.COLOR_BGR2GRAY)
    # Otsu 이진화 (배경이 흰색이므로 INV)
    _, step3_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    imwrite_korean(os.path.join(save_dirs['3'], filename), step3_binary)
    
    # --- [Step 4] Coarse Mask (모폴로지 연산) ---
    kernel = np.ones((7, 7), np.uint8)
    step4_closed = cv2.morphologyEx(step3_binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    imwrite_korean(os.path.join(save_dirs['4'], filename), step4_closed)
    
    # --- [Step 5] 최종 테두리 추출 ---
    contours, _ = cv2.findContours(step4_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    step5_final = step2_whitened.copy()
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        # 보라색 테두리 (두께 3)
        cv2.drawContours(step5_final, [main_contour], -1, (255, 0, 255), 3)
        # 중심점
        cv2.circle(step5_final, (cx, cy), 5, (0, 0, 255), -1)

    imwrite_korean(os.path.join(save_dirs['5'], filename), step5_final)

    print(f"✅ {filename} -> 0~5단계 저장 완료")

# ==========================================
# 5. 실행 로직
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Step-by-Step Analysis] 시작")
    print(f"📂 저장 경로: {output_root_folder}")
    
    # 단계별 저장 폴더 정의 및 생성
    step_folders = ['0_detection', '1_donut_mask', '2_whitened', '3_binary_raw', '4_coarse_mask', '5_final_result']
    save_dirs = {}
    
    for i, folder_name in enumerate(step_folders):
        # 딕셔너리에 '0', '1', ... 키로 경로 저장
        path = os.path.join(output_root_folder, folder_name)
        save_dirs[str(i)] = path
        os.makedirs(path, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue
        
        try:
            rel_path = os.path.relpath(root, input_folder)
        except:
            rel_path = os.path.basename(root)
            
        # 각 단계별 폴더 안에 원본 폴더 구조(27_30 등) 생성
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
            
            # 함수 실행 (경로 딕셔너리 전달)
            process_and_save_steps(img, file, current_save_dirs)

    print("\n✅ 모든 단계별 이미지 저장이 완료되었습니다. 12_coarseMask_contour 폴더를 확인하세요.")