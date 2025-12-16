import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.signal import find_peaks

# ==========================================
# 1. 설정 변수
# ==========================================
# [수정됨] 요청하신 새 모델 경로 반영
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\train_yolo8n_sizeup_boundingbox\weights\best.pt'

# 입력/출력 경로
input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\15_radial_profile_peaks'

CROP_SIZE = 1500

# [핵심 1] 동적 임계값 허용 오차 (Tolerance)
# 도넛 안에서 가장 어두운 값(톱니)을 찾으면, 거기서 +40 밝기까지만 톱니로 인정
DARKNESS_TOLERANCE = 40  

# [핵심 2] 도넛 마스킹 비율 (내부 회색 원 제거용)
# 바운딩 박스 반지름의 55% 안쪽은 아예 무시 (내부 회색 원 차단)
INNER_RATIO = 0.85  
OUTER_RATIO = 1.0   

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
# 3. 핵심 로직: Darkest Search + Radial Profile (Unroll)
# ==========================================
def process_radial_peaks(img, filename, save_dir):
    # 1. YOLO 추론
    results = model.predict(img, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0: 
        print(f"❌ {filename} -> YOLO 미검출")
        return

    # 2. 중심 및 반지름 계산
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    box_radius = min(x2 - x1, y2 - y1) // 2

    # 3. 크롭 (패딩 포함)
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

    # 4. 전처리 (CLAHE) - 명암비 극대화
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    _, _, v_ch = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_ch)

    # 5. [Step 1] 가장 어두운 톱니 영역 추출 (Dynamic Threshold)
    
    # (A) 검사할 도넛 마스크 생성
    mask_search = np.zeros_like(v_clahe)
    r_out = int(box_radius * OUTER_RATIO)
    r_in = int(box_radius * INNER_RATIO)
    cv2.circle(mask_search, center_crop, r_out, 255, -1) # 바깥 원
    cv2.circle(mask_search, center_crop, r_in, 0, -1)    # 안쪽 원 파내기

    # (B) 도넛 안에서 가장 어두운 값 찾기
    valid_pixels = v_clahe[mask_search > 0]
    if len(valid_pixels) == 0:
        print(f"⚠️ {filename} -> 유효 픽셀 없음")
        return

    min_val_in_donut = np.min(valid_pixels)
    
    # (C) 동적 임계값 설정 (최저값 + 오차범위)
    dynamic_thresh = min_val_in_donut + DARKNESS_TOLERANCE
    
    # (D) 이진화: 어두운 톱니만 흰색(255)으로 변환
    _, binary = cv2.threshold(v_clahe, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # 마스크 바깥쪽 노이즈 제거 (도넛 모양으로 자르기)
    binary = cv2.bitwise_and(binary, binary, mask=mask_search)

    # 6. [Step 2: 핵심] Radial Profile (이미지 펴기 & 신호 분석)
    
    # (A) 이미지 펴기 (Polar Transform: Unroll)
    # 원형 이미지를 직사각형 띠로 변환 (X축: 반경, Y축: 각도 -> 회전 후 X축: 각도)
    max_radius = CROP_SIZE // 2
    polar_img = cv2.linearPolar(binary, center_crop, max_radius, cv2.WARP_FILL_OUTLIERS)
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # (B) 프로파일 압축 (각도별 최대값)
    # 톱니가 있는 각도는 흰색(255), 없는 각도는 검은색(0)
    profile = np.max(polar_img, axis=0)
    
    # (C) Peak 찾기 (각도 기반)
    # distance=10: 톱니 간의 최소 각도 간격
    # width=5: 톱니의 최소 두께
    peaks, _ = find_peaks(profile, height=100, distance=10, width=5)
    
    count = len(peaks)
    
    # 7. 시각화 (좌표 역변환)
    result_img = cropped.copy()
    
    for angle_idx in peaks:
        # 그래프 상의 X축 인덱스(각도)를 다시 360도 라디안으로 변환
        angle_rad = (angle_idx / polar_img.shape[1]) * 2 * np.pi
        
        # 톱니의 중간 지점(반지름)에 점 찍기
        draw_radius = (r_in + r_out) // 2
        
        # 극좌표(r, theta) -> 직교좌표(x, y) 변환
        pt_x = int(center_crop[0] + draw_radius * np.cos(angle_rad))
        pt_y = int(center_crop[1] + draw_radius * np.sin(angle_rad))
        
        # 빨간 점 표시
        cv2.circle(result_img, (pt_x, pt_y), 6, (0, 0, 255), -1)

    # 범위 표시 (파란색)
    cv2.circle(result_img, center_crop, r_in, (255, 0, 0), 2)
    cv2.circle(result_img, center_crop, r_out, (255, 0, 0), 2)
    
    # 카운트 텍스트
    cv2.putText(result_img, f"Teeth: {count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # 저장
    imwrite_korean(os.path.join(save_dir, filename), result_img)
    print(f"✅ {filename} -> 개수: {count}")
    
    # [선택사항] 디버깅용 그래프 저장 (프로파일이 어떻게 생겼는지 확인 가능)
    # plt.figure(figsize=(10, 2))
    # plt.plot(profile)
    # plt.title(f"Radial Profile: {filename}")
    # plt.savefig(os.path.join(save_dir, f"{filename}_graph.png"))
    # plt.close()

# ==========================================
# 4. 실행
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Radial Profile Analysis] 시작")
    os.makedirs(output_root_folder, exist_ok=True)
    
    for root, dirs, files in os.walk(input_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue
        
        try:
            rel_path = os.path.relpath(root, input_folder)
        except:
            rel_path = os.path.basename(root)
            
        save_path = os.path.join(output_root_folder, rel_path)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n📂 처리 중: {rel_path}")
        
        for file in bmp_files:
            img_path = os.path.join(root, file)
            img = imread_korean(img_path)
            if img is None: continue
            
            process_radial_peaks(img, file, save_path)

    print("\n✅ 완료. 15_radial_profile_peaks 폴더를 확인하세요.")