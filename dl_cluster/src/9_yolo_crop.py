import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# 1. 설정 변수
# ==========================================
# 절대 경로 사용 (특수문자 오류 방지)
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\runs\detect\gear_train_v13\weights\best.pt'

# 입력 데이터 폴더
input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'

# 결과 저장 최상위 폴더
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\9_yolo_crop'

# 하위 폴더 경로 정의
results_folder = os.path.join(output_root_folder, 'results') # 크롭 이미지 저장
box_folder = os.path.join(output_root_folder, 'box')         # 원본 + 박스 이미지 저장

# 크롭할 이미지 크기
CROP_SIZE = 1500

# ==========================================
# 2. YOLO 모델 로드
# ==========================================
print(f"🔄 모델 로딩 중: {model_path}")
try:
    model = YOLO(model_path)
    print("✅ 모델 로드 성공!")
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
# 4. YOLO 기반 센터 크롭 및 정보 추출 함수
# ==========================================
def yolo_center_crop_and_info(img, filename):
    """
    YOLO로 톱니바퀴를 찾아 크롭된 이미지와 박스 정보를 함께 반환합니다.
    반환값: (cropped_img, (x1, y1, x2, y2, cx, cy))
    """
    # YOLO 추론
    results = model.predict(img, conf=0.5, verbose=False)
    
    if len(results[0].boxes) == 0:
        print(f"❌ {filename} -> 톱니바퀴 못 찾음")
        return None, None

    # 박스 좌표 가져오기
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    
    # 중심점 계산
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    
    print(f"✅ {filename} -> 중심 발견: ({cx}, {cy})")
    
    # 박스 정보 튜플 생성 (원본 좌표 기준)
    box_info = (x1, y1, x2, y2, cx, cy)

    # --- 크롭 로직 (results 폴더용) ---
    h, w = img.shape[:2]
    half = CROP_SIZE // 2
    
    src_x1, src_y1 = cx - half, cy - half
    src_x2, src_y2 = cx + half, cy + half
    
    # 패딩 계산
    pad_l = abs(src_x1) if src_x1 < 0 else 0
    pad_t = abs(src_y1) if src_y1 < 0 else 0
    pad_r = (src_x2 - w) if src_x2 > w else 0
    pad_b = (src_y2 - h) if src_y2 > h else 0
    
    # 패딩 적용
    if any([pad_l, pad_t, pad_r, pad_b]):
        img_padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, 
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
        src_x1 += pad_l; src_x2 += pad_l
        src_y1 += pad_t; src_y2 += pad_t
    else:
        img_padded = img.copy()
        
    # 최종 크롭
    cropped = img_padded[src_y1:src_y2, src_x1:src_x2]
    
    if cropped.shape[:2] != (CROP_SIZE, CROP_SIZE):
        cropped = cv2.resize(cropped, (CROP_SIZE, CROP_SIZE))
        
    return cropped, box_info

# ==========================================
# 5. 실행 로직
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [YOLO Crop & Visualize] 시작")
    print(f"📂 결과 저장 루트: {output_root_folder}")
    print(f"   L results: 크롭된 이미지 (1300x1300)")
    print(f"   L box: 원본 이미지 + 바운딩 박스/중심점 표시 (크롭 X)")

    # 폴더 생성
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(box_folder, exist_ok=True)
    
    for root, dirs, files in os.walk(input_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue
        
        try:
            rel_path = os.path.relpath(root, input_folder)
        except:
            rel_path = os.path.basename(root)
        
        # 하위 폴더 구조 생성
        save_path_results = os.path.join(results_folder, rel_path)
        save_path_box = os.path.join(box_folder, rel_path)
        os.makedirs(save_path_results, exist_ok=True)
        os.makedirs(save_path_box, exist_ok=True)
        
        print(f"\n📂 폴더 처리 중: {rel_path}")
        
        for file in bmp_files:
            img_path = os.path.join(root, file)
            img = imread_korean(img_path)
            if img is None: continue
            
            # 1. YOLO로 크롭 이미지와 박스 정보 받아오기
            cropped, box_info = yolo_center_crop_and_info(img, file)
            
            if cropped is not None and box_info is not None:
                # --- 2-1. results 폴더에 크롭 이미지 저장 ---
                save_file_path_res = os.path.join(save_path_results, file)
                imwrite_korean(save_file_path_res, cropped)

                # --- 2-2. box 폴더에 원본 시각화 이미지 저장 ---
                x1, y1, x2, y2, cx, cy = box_info
                
                # [핵심] 여기서 원본 이미지(img)를 복사해서 사용하므로 크롭되지 않습니다.
                box_img = img.copy() 
                
                # 바운딩 박스 그리기 (초록색, 두께 3)
                cv2.rectangle(box_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # 중심점 그리기 (빨간색, 반지름 10, 꽉 채움)
                cv2.circle(box_img, (cx, cy), 10, (0, 0, 255), -1)
                
                save_file_path_box = os.path.join(save_path_box, file)
                imwrite_korean(save_file_path_box, box_img)

    print("\n✅ 모든 작업이 완료되었습니다! 9_yolo_crop 폴더를 확인하세요.")