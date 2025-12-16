import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.signal import find_peaks

# ==========================================
# 1. 설정 변수
# ==========================================
# [고정] 사용자 지정 모델 경로
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\train_yolo8n_sizeup_boundingbox\weights\best.pt'

# 입력 데이터 폴더
input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'

# [수정됨] 결과 저장 폴더 (15_1 버전: 업그레이드)
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\15_1_radial_profile_peaks'

CROP_SIZE = 1500

# [핵심 파라미터]
# 1. 도넛 마스킹: 내부 회색 원 제거 (60% 파냄)
INNER_RATIO = 0.85
OUTER_RATIO = 0.99

# 2. 노이즈 제거: 이 크기보다 작은 덩어리는 톱니로 보지 않음
MIN_TOOTH_AREA = 40

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
# 3. 핵심 로직: Adaptive Threshold + Radial Profile
# ==========================================
def process_radial_peaks_v1_1(img, filename, save_dirs):
    # --- 1. YOLO 추론 ---
    results = model.predict(img, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0:
        print(f"❌ {filename} -> YOLO 미검출")
        return

    # 중심 및 반지름 계산
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    box_radius = min(x2 - x1, y2 - y1) // 2

    # --- 2. 크롭 (패딩 포함) ---
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

    # --- 3. 전처리 (CLAHE + Adaptive Threshold) ---
    # [개선] 적응형 이진화로 '뭉개짐' 해결
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    _, _, v_ch = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_ch)

    # 적응형 이진화: 주변보다 어두운 부분(톱니)을 흰색(255)으로 반전
    binary = cv2.adaptiveThreshold(
        v_clahe, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 5
    )
    
    imwrite_korean(os.path.join(save_dirs['1_binary_raw'], filename), binary)

    # --- 4. 도넛 마스킹 & 노이즈 제거 ---
    mask_search = np.zeros_like(binary)
    r_out = int(box_radius * OUTER_RATIO)
    r_in = int(box_radius * INNER_RATIO)
    
    cv2.circle(mask_search, center_crop, r_out, 255, -1) # 바깥 원
    cv2.circle(mask_search, center_crop, r_in, 0, -1)    # 안쪽 원 파내기
    
    # 마스크 적용
    binary_masked = cv2.bitwise_and(binary, binary, mask=mask_search)
    
    # 모폴로지 (자잘한 노이즈 제거)
    kernel = np.ones((3, 3), np.uint8)
    binary_clean = cv2.morphologyEx(binary_masked, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # --- 5. [중요] 컨투어 필터링 (엉뚱한 곳 카운팅 방지) ---
    # 여기서 너무 작은 점이나 엉뚱한 덩어리를 걸러냅니다.
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 필터링된 깨끗한 이미지를 새로 그림 (분석용)
    binary_final = np.zeros_like(binary_clean)
    
    # 시각화용 이미지 (테두리 확인용)
    contour_view = cropped.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_TOOTH_AREA:  # 일정 크기 이상인 것만 '진짜 톱니'로 인정
            cv2.drawContours(binary_final, [cnt], -1, 255, -1) # 흰색으로 채움 (Radial Profile 분석용)
            cv2.drawContours(contour_view, [cnt], -1, (0, 255, 0), 2) # 초록색 테두리 (시각화용)

    imwrite_korean(os.path.join(save_dirs['2_binary_filtered'], filename), binary_final)
    imwrite_korean(os.path.join(save_dirs['3_contours'], filename), contour_view)

    # --- 6. Radial Profile Analysis (이미지 펴기) ---
    max_radius = CROP_SIZE // 2
    # 필터링된 깨끗한 binary_final 이미지를 사용
    polar_img = cv2.linearPolar(binary_final, center_crop, max_radius, cv2.WARP_FILL_OUTLIERS)
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 프로파일 압축 (각도별 최대값)
    profile = np.max(polar_img, axis=0)
    
    # Peak 찾기
    peaks, _ = find_peaks(profile, height=100, distance=10, width=3)
    count = len(peaks)
    
    # --- 7. 최종 결과 시각화 ---
    final_img = cropped.copy()
    
    # (A) 필터링된 테두리 그리기 (요청사항 1번 해결)
    # 위에서 구한 유효 컨투어를 다시 그림
    contours_final, _ = cv2.findContours(binary_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final_img, contours_final, -1, (0, 255, 0), 2) # 초록색 실선
    
    # (B) 찾은 산(Peak) 위치에 점 찍기
    for angle_idx in peaks:
        angle_rad = (angle_idx / polar_img.shape[1]) * 2 * np.pi
        
        # 톱니의 중간 지점에 점 찍기
        draw_radius = (r_in + r_out) // 2
        pt_x = int(center_crop[0] + draw_radius * np.cos(angle_rad))
        pt_y = int(center_crop[1] + draw_radius * np.sin(angle_rad))
        
        # 빨간 점 표시 (요청사항 2번 해결 - 엉뚱한 곳 안 찍힘)
        cv2.circle(final_img, (pt_x, pt_y), 6, (0, 0, 255), -1)

    # 범위 표시 (파란색)
    cv2.circle(final_img, center_crop, r_in, (255, 0, 0), 2)
    cv2.circle(final_img, center_crop, r_out, (255, 0, 0), 2)
    
    cv2.putText(final_img, f"Count: {count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # 저장
    imwrite_korean(os.path.join(save_dirs['4_final_result'], filename), final_img)
    print(f"✅ {filename} -> 개수: {count}")

# ==========================================
# 4. 실행 로직
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Radial Profile Analysis v1.1] 시작")
    
    # 단계별 저장 폴더
    step_folders = ['0_crop', '1_binary_raw', '2_binary_filtered', '3_contours', '4_final_result']
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
            
        # 하위 폴더 생성
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
            
            process_radial_peaks_v1_1(img, file, current_save_dirs)

    print("\n✅ 완료. 15_1_radial_profile_peaks 폴더를 확인하세요.")