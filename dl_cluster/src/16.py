import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.signal import find_peaks

# ==========================================
# 1. 설정 변수
# ==========================================
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\train_yolo8n_sizeup_boundingbox\weights\best.pt'
input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\16_final_radial_solution'

CROP_SIZE = 1500

# [해결책 1] 분석할 도넛 구간 설정 (Unroll 이미지 기준)
# 회색 원을 피하기 위해 MIN을 높이고(0.65), 톱니 끝을 포함하기 위해 MAX를 충분히 줌(0.95)
RADIUS_MIN_RATIO = 0.84
RADIUS_MAX_RATIO = 1 

# [해결책 2] 신호 분석 민감도
PEAK_HEIGHT = 50      # 그래프 높이 기준
PEAK_DISTANCE = 10    # 톱니 간 최소 간격

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

def imread_korean(file_path):
    try:
        img_array = np.fromfile(file_path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_korean(filename, img):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        return False
    except Exception:
        return False

# ==========================================
# 3. 핵심 로직: Polar Transform & Signal Analysis
# ==========================================
def process_gear_final(img, filename, save_dirs):
    # 1. YOLO 추론
    results = model.predict(img, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0:
        print(f"❌ {filename} -> YOLO 미검출")
        return

    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    # 박스 반지름 (조금 여유있게 잡을 필요 없음, 비율로 조절할 것이므로)
    box_radius = min(x2 - x1, y2 - y1) // 2

    # 2. 크롭
    h, w = img.shape[:2]
    half = CROP_SIZE // 2
    pad_l = abs(cx - half) if (cx - half) < 0 else 0
    pad_t = abs(cy - half) if (cy - half) < 0 else 0
    pad_r = (cx + half - w) if (cx + half) > w else 0
    pad_b = (cy + half - h) if (cy + half) > h else 0
    
    if any([pad_l, pad_t, pad_r, pad_b]):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cx += pad_l; cy += pad_t
    
    src_x1, src_y1 = cx - half, cy - half
    src_x2, src_y2 = cx + half, cy + half
    cropped = img[src_y1:src_y2, src_x1:src_x2].copy()
    center_crop = (CROP_SIZE // 2, CROP_SIZE // 2)
    
    imwrite_korean(os.path.join(save_dirs['0_crop'], filename), cropped)

    # 3. Polar Transform (이미지 펴기)
    max_radius = CROP_SIZE // 2
    polar_img = cv2.linearPolar(cropped, center_crop, max_radius, cv2.WARP_FILL_OUTLIERS)
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 가로: 반지름, 세로: 각도

    # 4. [해결책 1] 유효 구간 자르기 (회색 원 제거)
    # YOLO 박스 반지름 기준으로 비율 적용
    valid_start = int(box_radius * RADIUS_MIN_RATIO)
    valid_end = int(box_radius * RADIUS_MAX_RATIO)
    
    # 범위 보정
    valid_start = max(0, valid_start)
    valid_end = min(polar_img.shape[1], valid_end)
    
    strip_roi = polar_img[:, valid_start:valid_end]
    
    imwrite_korean(os.path.join(save_dirs['1_unrolled'], filename), strip_roi)

    # 5. 전처리 & 이진화
    gray_strip = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_strip = clahe.apply(gray_strip)
    
    # 적응형 이진화 (회색 원이 있어도 국소적으로 어두운 톱니만 잡음)
    binary_strip = cv2.adaptiveThreshold(
        enhanced_strip, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 5
    )
    
    # 모폴로지 (끊어진 톱니 연결)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)) # 세로로 긴 커널
    binary_strip = cv2.morphologyEx(binary_strip, cv2.MORPH_CLOSE, kernel, iterations=2)

    imwrite_korean(os.path.join(save_dirs['2_binary'], filename), binary_strip)

    # 6. [해결책 2] 신호 분석 (1D Signal)
    # 가로 방향(반지름)으로 압축 -> 세로 방향(각도) 그래프 생성
    signal = np.max(binary_strip, axis=1) # max를 쓰면 하나라도 톱니가 있으면 신호가 뜸
    
    # Peak 찾기
    peaks, _ = find_peaks(signal, height=PEAK_HEIGHT, distance=PEAK_DISTANCE)
    count = len(peaks)

    # 7. 시각화 (Unroll 이미지)
    vis_strip = strip_roi.copy()
    for p in peaks:
        cv2.line(vis_strip, (0, p), (vis_strip.shape[1], p), (0, 0, 255), 2)
    
    imwrite_korean(os.path.join(save_dirs['3_signal_vis'], filename), vis_strip)

    # 8. 최종 결과 (원본 복원)
    final_img = cropped.copy()
    draw_radius = (valid_start + valid_end) // 2
    
    for p in peaks:
        # 각도 변환 (인덱스 -> 라디안)
        angle_rad = (p / polar_img.shape[0]) * 2 * np.pi
        
        pt_x = int(center_crop[0] + draw_radius * np.cos(angle_rad))
        pt_y = int(center_crop[1] + draw_radius * np.sin(angle_rad))
        
        cv2.circle(final_img, (pt_x, pt_y), 6, (0, 0, 255), -1)

    # 범위 표시
    cv2.circle(final_img, center_crop, valid_start, (255, 0, 0), 2)
    cv2.circle(final_img, center_crop, valid_end, (255, 0, 0), 2)
    
    cv2.putText(final_img, f"Count: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    
    imwrite_korean(os.path.join(save_dirs['4_final_result'], filename), final_img)
    print(f"✅ {filename} -> 개수: {count}")

# ==========================================
# 4. 실행
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Final Solution: Radial Profile] 시작")
    
    step_folders = ['0_crop', '1_unrolled', '2_binary', '3_signal_vis', '4_final_result']
    save_dirs = {}
    for folder in step_folders:
        path = os.path.join(output_root_folder, folder)
        save_dirs[folder] = path
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
        
        print(f"\n📂 처리 중: {rel_path}")
        
        for file in bmp_files:
            img_path = os.path.join(root, file)
            img = imread_korean(img_path)
            if img is None: continue
            
            process_gear_final(img, file, current_save_dirs)

    print("\n✅ 완료. 16_final_radial_solution 폴더 확인.")