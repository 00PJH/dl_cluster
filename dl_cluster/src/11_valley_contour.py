import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# 1. 설정 변수
# ==========================================
# 학습된 모델 경로
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\train_yolo8n_sizeup_boundingbox\weights\best.pt'
input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\11_valley_counting'

# 크롭 사이즈
CROP_SIZE = 1500

# [중요] 도넛 마스킹 설정
# 골짜기를 찾기 위해 톱니 뿌리 부분은 확실히 보여야 합니다.
OUTER_RADIUS = 625  # 톱니 바깥쪽 (배경 제거용)
INNER_RADIUS = 400  # 톱니 안쪽 (구멍 제거용)

# 골짜기로 인정할 최소 깊이 (픽셀)
# 고무줄(Hull)에서 얼마나 안쪽으로 파여야 골짜기로 칠 것인가?
MIN_DEFECT_DEPTH = 10 

# ==========================================
# 2. 모델 로드 및 유틸리티
# ==========================================
print(f"🔄 모델 로딩: {model_path}")
try:
    model = YOLO(model_path)
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 실패: {e}")
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
# 3. 핵심 로직: YOLO -> 도넛 -> 골짜기(Valley) 추출
# ==========================================
def process_gear_valley(img, filename):
    # 1. YOLO 추론 및 중심 잡기
    results = model.predict(img, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0:
        print(f"❌ {filename} -> YOLO 미검출")
        return None

    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    # 2. 크롭 (패딩 포함)
    h, w = img.shape[:2]
    half = CROP_SIZE // 2
    
    pad_l = abs(cx - half) if (cx - half) < 0 else 0
    pad_t = abs(cy - half) if (cy - half) < 0 else 0
    pad_r = (cx + half - w) if (cx + half) > w else 0
    pad_b = (cy + half - h) if (cy + half) > h else 0
    
    if any([pad_l, pad_t, pad_r, pad_b]):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, 
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cx += pad_l; cy += pad_t

    src_x1, src_y1 = cx - half, cy - half
    src_x2, src_y2 = cx + half, cy + half
    cropped = img[src_y1:src_y2, src_x1:src_x2].copy()
    
    center_crop = (CROP_SIZE // 2, CROP_SIZE // 2)

    # 3. 전처리 (도넛 마스킹)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    
    # 명암비 향상 (CLAHE) - 톱니 경계 뚜렷하게
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    
    # 마스크 생성
    mask = np.zeros_like(v_clahe)
    cv2.circle(mask, center_crop, OUTER_RADIUS, 255, -1)
    cv2.circle(mask, center_crop, INNER_RADIUS, 0, -1)
    masked_img = cv2.bitwise_and(v_clahe, v_clahe, mask=mask)
    
    # 이진화 (Otsu)
    _, binary = cv2.threshold(masked_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 노이즈 제거 (Morphology)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. 외곽선 및 Hull 추출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return cropped
    
    main_contour = max(contours, key=cv2.contourArea)
    
    # [핵심] Convex Hull (인덱스 반환 모드)
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    
    # [핵심] Convexity Defects (골짜기 찾기)
    # defects 구조: [start_index, end_index, farthest_pt_index, fixpt_depth]
    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
    except:
        return cropped # 에러 시 원본 반환

    valleys = []
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            
            # d는 거리(depth)인데 256을 나눠야 실제 픽셀 거리임
            depth = d / 256.0
            
            # 깊이가 너무 얕으면(그냥 표면의 거친 부분) 무시
            if depth > MIN_DEFECT_DEPTH:
                # f가 골짜기(Valley)의 좌표 인덱스
                far_pt = tuple(main_contour[f][0])
                
                # 추가 검증: 골짜기 위치가 도넛 범위 안에 있는지
                dist_from_center = np.linalg.norm(np.array(far_pt) - np.array(center_crop))
                if INNER_RADIUS < dist_from_center < OUTER_RADIUS:
                    valleys.append(far_pt)

    # 5. 시각화
    result_img = cropped.copy()
    
    # (A) 찾은 외곽선 (보라색)
    cv2.drawContours(result_img, [main_contour], -1, (255, 0, 255), 2)
    
    # (B) Hull(고무줄) 그리기 (초록색 점선 느낌)
    # hull_points = cv2.convexHull(main_contour, returnPoints=True)
    # cv2.drawContours(result_img, [hull_points], -1, (0, 255, 0), 1)
    
    # (C) 골짜기(Valley) 포인트 (파란점)
    # 사다리꼴 톱니 사이의 '오목한 곳'을 셉니다.
    for pt in valleys:
        cv2.circle(result_img, pt, 8, (255, 0, 0), -1) # 파란색 점
        
    count = len(valleys)
    cv2.putText(result_img, f"Teeth: {count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4) # 노란색 글씨
    
    # 마스킹 범위 표시 (참고용)
    cv2.circle(result_img, center_crop, OUTER_RADIUS, (0, 255, 0), 1)
    cv2.circle(result_img, center_crop, INNER_RADIUS, (0, 255, 0), 1)
    
    print(f"✅ {filename} -> 골짜기(Valley) 개수: {count}")
    return result_img

# ==========================================
# 4. 실행 로직
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Valley Counting] 사다리꼴 톱니 분석 시작")
    os.makedirs(output_folder, exist_ok=True)
    
    for root, dirs, files in os.walk(input_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue
        
        try:
            rel_path = os.path.relpath(root, input_folder)
        except:
            rel_path = os.path.basename(root)
            
        save_path = os.path.join(output_folder, rel_path)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n📂 폴더: {rel_path}")
        
        for file in bmp_files:
            img_path = os.path.join(root, file)
            img = imread_korean(img_path)
            if img is None: continue
            
            result_img = process_gear_valley(img, file)
            
            if result_img is not None:
                save_file = os.path.join(save_path, file)
                imwrite_korean(save_file, result_img)

    print("\n✅ 분석 완료. 12_valley_counting 폴더를 확인하세요.")