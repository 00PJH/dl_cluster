import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.signal import find_peaks

# ==========================================
# 1. 환경 설정
# ==========================================
# [필수] 요청하신 고정 모델 경로
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\train_yolo8n_sizeup_boundingbox\weights\best.pt'

input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\18_polar_signal_analysis'

CROP_SIZE = 1500

# [핵심 파라미터]
# 톱니바퀴를 폈을 때, 유효한 반지름 구간 (0.0 ~ 1.0)
# 바깥쪽 회색 원을 피하기 위해 MAX를 0.95 정도로 줄임
# 안쪽 뭉개짐을 피하기 위해 MIN을 0.65 정도로 높임
RADIUS_MIN_RATIO = 0.85  
RADIUS_MAX_RATIO = 1  

# 톱니 감지 민감도 (그래프 높이)
# 값이 낮으면 희미한 톱니도 잡고, 높으면 선명한 것만 잡음
PEAK_HEIGHT_THRESHOLD = 40

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
# 3. 핵심 알고리즘: Polar Transform & Signal Analysis
# ==========================================
def analyze_gear_polar(img, filename, save_dirs):
    # --- 1. YOLO Detection ---
    results = model.predict(img, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0:
        print(f"❌ {filename} -> YOLO 미검출")
        return None

    # 중심 좌표 및 반지름 계산
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    # 기어의 반지름 추정 (박스 크기의 절반)
    gear_radius = min(x2 - x1, y2 - y1) // 2

    # --- 2. Safe Crop (1500x1500) ---
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

    # --- 3. [핵심] Polar Transform (이미지 펴기) ---
    # 원형 이미지를 직사각형 띠(Strip)로 변환
    # X축: 각도(Angle), Y축: 거리(Radius)
    max_radius = CROP_SIZE // 2
    # cv2.WARP_FILL_OUTLIERS: 빈 공간 보간
    polar_img = cv2.linearPolar(cropped, center_crop, max_radius, cv2.WARP_FILL_OUTLIERS)
    
    # 보기 편하게 90도 회전 (위쪽이 바깥, 아래쪽이 안쪽이 됨 -> 가로축이 각도가 됨)
    # 회전 후: 가로(Width)=반지름, 세로(Height)=각도
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # --- 4. Band Selection (회색 원 & 안쪽 제거) ---
    # 회전된 이미지에서 "가로(X)축"이 반지름(Radius)입니다.
    # 유효한 반지름 구간만 잘라냅니다.
    h_pol, w_pol = polar_img.shape[:2]
    
    # 0 ~ w_pol 범위 중 우리가 원하는 구간 계산
    # YOLO 박스 반지름(gear_radius) 기준으로 비율 적용
    valid_start = int(gear_radius * RADIUS_MIN_RATIO)
    valid_end = int(gear_radius * RADIUS_MAX_RATIO)
    
    # 인덱스 범위 초과 방지
    valid_start = max(0, valid_start)
    valid_end = min(w_pol, valid_end)
    
    # 띠(Strip) 잘라내기 [세로(각도):전체, 가로(반지름):유효구간]
    strip_roi = polar_img[:, valid_start:valid_end]
    
    imwrite_korean(os.path.join(save_dirs['1_unrolled_roi'], filename), strip_roi)

    # --- 5. Preprocessing & Binarization ---
    # CLAHE 적용 (명암비 극대화)
    gray_strip = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_strip = clahe.apply(gray_strip)
    
    # 적응형 이진화 (Adaptive Threshold)
    # 그림자가 져도 국소적으로 어두우면 잡아냄
    binary_strip = cv2.adaptiveThreshold(
        enhanced_strip, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, # 톱니(어두움)를 흰색으로
        25, 5
    )
    
    # 모폴로지 (세로 방향으로 찢어진 톱니 연결)
    # 커널을 세로로 길게 쓰면(1, 3) 끊어진 톱니 연결에 좋음
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary_strip = cv2.morphologyEx(binary_strip, cv2.MORPH_CLOSE, kernel, iterations=2)

    imwrite_korean(os.path.join(save_dirs['2_binary_strip'], filename), binary_strip)

    # --- 6. 1D Signal Analysis (신호 분석) ---
    # 가로(Radius) 방향으로 픽셀 값을 합칩니다.
    # 결과는 세로(Angle) 방향의 1차원 그래프가 됩니다.
    # axis=1 (가로) 방향으로 평균/최대값 추출
    signal = np.mean(binary_strip, axis=1)
    
    # Peak 찾기 (톱니 위치 검출)
    # distance: 톱니 간 최소 간격 (픽셀 단위, 해상도에 따라 조절)
    peaks, _ = find_peaks(signal, height=PEAK_HEIGHT_THRESHOLD, distance=15)
    
    count = len(peaks)

    # --- 7. 시각화 (Unroll 이미지에 표시) ---
    result_strip = strip_roi.copy()
    for p in peaks:
        # 피크 위치(Y좌표)에 가로선을 그어 표시
        cv2.line(result_strip, (0, p), (result_strip.shape[1], p), (0, 0, 255), 2)
        
    imwrite_korean(os.path.join(save_dirs['3_result_strip'], filename), result_strip)

    # --- 8. 최종 결과 (원본 좌표 복원) ---
    final_img = cropped.copy()
    
    # 복원용 반지름 (그리기 예쁘게 중간쯤)
    draw_radius = (valid_start + valid_end) // 2
    
    for p in peaks:
        # Unrolled 이미지의 Y좌표(p) -> 360도 각도로 변환
        # 전체 높이(h_pol)가 360도에 해당함
        angle_rad = (p / h_pol) * 2 * np.pi
        
        # 극좌표 -> 직교좌표
        pt_x = int(center_crop[0] + draw_radius * np.cos(angle_rad))
        pt_y = int(center_crop[1] + draw_radius * np.sin(angle_rad))
        
        cv2.circle(final_img, (pt_x, pt_y), 6, (0, 0, 255), -1)

    # 범위 표시 (파란원)
    cv2.circle(final_img, center_crop, valid_start, (255, 0, 0), 2)
    cv2.circle(final_img, center_crop, valid_end, (255, 0, 0), 2)
    
    cv2.putText(final_img, f"Count: {count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    
    imwrite_korean(os.path.join(save_dirs['4_final_circle'], filename), final_img)
    
    print(f"✅ {filename} -> 톱니 개수: {count}")
    return count

# ==========================================
# 4. 실행 로직
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Polar Signal Analysis] 시작")
    
    # 단계별 폴더 생성
    step_folders = ['0_crop', '1_unrolled_roi', '2_binary_strip', '3_result_strip', '4_final_circle']
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
        
        print(f"\n📂 폴더 처리 중: {rel_path}")
        
        for file in bmp_files:
            img_path = os.path.join(root, file)
            img = imread_korean(img_path)
            if img is None: continue
            
            analyze_gear_polar(img, file, current_save_dirs)

    print("\n✅ 완료. 18_polar_signal_analysis 폴더를 확인하세요.")