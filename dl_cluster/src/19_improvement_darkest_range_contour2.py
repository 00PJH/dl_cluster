import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO

# ==========================================
# 1. 설정 변수
# ==========================================
# 모델 경로
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\train_yolo8n_sizeup_boundingbox\weights\best.pt'

# 입력 및 출력 폴더
input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\19_improvement_darkest_range_contour2'

# CSV 저장 경로
csv_save_path = os.path.join(output_root_folder, 'accuracy_report.csv')

CROP_SIZE = 1500

# [핵심 파라미터 수정]
# 1. 도넛 마스킹 비율 (내부 잡음 완벽 차단)
# 0.83: 바운딩 박스 반지름의 83% 지점까지 구멍을 뚫어버립니다. (아주 타이트함)
INNER_RATIO = 0.85
OUTER_RATIO = 1.0

# 2. 가장 어두운 영역 추출 허용 오차
# 값이 클수록(60) 회색도 포함, 작을수록(30) 아주 검은 것만 포함
DARKNESS_TOLERANCE = 65

# 3. 노이즈 제거 (컨투어 최소 면적)
MIN_TOOTH_AREA = 50 

# ==========================================
# 2. 모델 로드 및 유틸리티
# ==========================================
print(f"🔄 모델 로딩: {model_path}")
try:
    model = YOLO(model_path)
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

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

def imread_korean(file_path):
    try:
        img_array = np.fromfile(file_path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

# ==========================================
# 3. 핵심 로직
# ==========================================
def process_gear_improvement(img, filename, save_dirs):
    # --- 1. YOLO 추론 & 크롭 ---
    results = model.predict(img, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0: return None

    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    # 박스 반지름 (짧은 쪽 기준)
    box_radius = min(x2 - x1, y2 - y1) // 2

    # 크롭 (패딩 포함)
    h, w = img.shape[:2]
    half = CROP_SIZE // 2
    
    # 패딩 계산 및 적용
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

    # --- 2. 도넛 마스크 생성 (영역 제한의 핵심) ---
    mask_donut = np.zeros(cropped.shape[:2], dtype=np.uint8)
    
    # 유효 반경 계산
    r_out = int(box_radius * OUTER_RATIO)
    r_in = int(box_radius * INNER_RATIO)
    
    # 도넛 그리기 (흰색 영역만 분석 대상)
    cv2.circle(mask_donut, center_crop, r_out, 255, -1)
    cv2.circle(mask_donut, center_crop, r_in, 0, -1)

    # --- 3. 도넛 영역 한정 전처리 ---
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    _, _, v_ch = cv2.split(hsv)
    
    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_ch)
    
    imwrite_korean(os.path.join(save_dirs['1_clahe'], filename), v_clahe)

    # --- 4. Darkest Region Extraction ---
    # 도넛 마스크 내의 픽셀값만 가져옴 (중요: 다른 영역의 어두운 값은 무시됨)
    valid_pixels = v_clahe[mask_donut > 0]
    
    if len(valid_pixels) == 0: return None

    # 가장 어두운 값(Min) 찾기
    min_val = np.min(valid_pixels)
    
    # 동적 임계값 설정
    dynamic_thresh = min_val + DARKNESS_TOLERANCE
    
    # 이진화: 어두운 톱니를 흰색(255)으로
    _, binary = cv2.threshold(v_clahe, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # [핵심] 도넛 마스크 적용 (영역 밖은 강제로 0으로 만듦)
    binary_masked = cv2.bitwise_and(binary, binary, mask=mask_donut)

    imwrite_korean(os.path.join(save_dirs['2_binary'], filename), binary_masked)

    # --- 5. 모폴로지 ---
    kernel = np.ones((5, 5), np.uint8)
    binary_clean = cv2.morphologyEx(binary_masked, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    imwrite_korean(os.path.join(save_dirs['3_morphology'], filename), binary_clean)

    # --- 6. 테두리 추출 및 카운팅 (이중 안전장치) ---
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_img = cropped.copy()
    teeth_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_TOOTH_AREA:
            # 중심점 계산
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx_t = int(M["m10"] / M["m00"])
                cy_t = int(M["m01"] / M["m00"])
                
                # [안전장치] 중심점 거리 검사
                # 마스크가 있더라도 혹시 모를 노이즈를 위해 거리로 한 번 더 거름
                dist_from_center = np.sqrt((cx_t - center_crop[0])**2 + (cy_t - center_crop[1])**2)
                
                # 거리가 내부 원보다 커야 함 (안쪽 잡음 배제)
                if dist_from_center >= r_in:
                    teeth_count += 1
                    # 테두리 그리기
                    cv2.drawContours(final_img, [cnt], -1, (0, 255, 0), 2)
                    # 중심점 찍기
                    cv2.circle(final_img, (cx_t, cy_t), 5, (0, 0, 255), -1)

    # 마스크 범위 표시 (파란색 원)
    cv2.circle(final_img, center_crop, r_in, (255, 0, 0), 2)
    cv2.circle(final_img, center_crop, r_out, (255, 0, 0), 2)

    cv2.putText(final_img, f"Count: {teeth_count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    imwrite_korean(os.path.join(save_dirs['4_final_result'], filename), final_img)
    
    return teeth_count

# ==========================================
# 4. 실행 및 리포트 생성
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Improved Darkest Range Contour - Strict Area] 시작")
    os.makedirs(output_root_folder, exist_ok=True)
    
    step_folders = ['0_crop', '1_clahe', '2_binary', '3_morphology', '4_final_result']
    save_dirs = {}
    for folder in step_folders:
        path = os.path.join(output_root_folder, folder)
        save_dirs[folder] = path
        os.makedirs(path, exist_ok=True)

    results_list = []

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
            
            pred_count = process_gear_improvement(img, file, current_save_dirs)
            
            if pred_count is not None:
                # 정답 파싱
                try:
                    gt_str = file.split('_')[0]
                    gt_count = int(gt_str)
                    is_correct = (pred_count == gt_count)
                except:
                    gt_count = -1
                    is_correct = False
                
                print(f"  - {file}: 정답={gt_count}, 예측={pred_count} -> {'O' if is_correct else 'X'}")
                
                results_list.append({
                    'Folder': rel_path,
                    'Filename': file,
                    'Ground_Truth': gt_count,
                    'Predicted': pred_count,
                    'Correct': is_correct
                })

    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
        
        valid_df = df[df['Ground_Truth'] != -1]
        
        if not valid_df.empty:
            print("\n📊 [정확도 분석 결과]")
            accuracy_report = valid_df.groupby('Ground_Truth').apply(
                lambda x: pd.Series({
                    'Total': len(x),
                    'Correct': x['Correct'].sum(),
                    'Accuracy(%)': (x['Correct'].sum() / len(x)) * 100
                })
            ).reset_index()
            
            total_acc = (valid_df['Correct'].sum() / len(valid_df)) * 100
            print(accuracy_report)
            print(f"\n🏆 최종 종합 정확도: {total_acc:.2f}%")
            
            summary_path = os.path.join(output_root_folder, 'accuracy_summary.csv')
            accuracy_report.to_csv(summary_path, index=False, encoding='utf-8-sig')

    print("\n✅ 완료. 19_improvement_darkest_range_contour2 폴더를 확인하세요.")