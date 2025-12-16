import cv2
import numpy as np
import os
import math
import pandas as pd
from ultralytics import YOLO

# ==========================================
# 1. 설정 변수
# ==========================================
# 모델 경로
model_path = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\src\train_yolo8n_sizeup_boundingbox\weights\best.pt'

# 입력 및 출력 폴더
input_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\17_convexity_defects_analysis'

# CSV 저장 경로
csv_save_path = os.path.join(output_root_folder, 'accuracy_report.csv')

CROP_SIZE = 1500

# [핵심 설정]
# 도넛 마스킹 비율 (이 안쪽/바깥쪽은 아예 무시)
INNER_RATIO = 0.85  
OUTER_RATIO = 1.0   

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
# 3. 핵심 로직: 적응형 이진화 + 볼록 결함(Defects)
# ==========================================
def process_gear_defects(img, filename, save_dirs):
    # --- 1. YOLO 추론 및 크롭 ---
    results = model.predict(img, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0: 
        print(f"❌ {filename} -> YOLO 미검출")
        return None

    # 박스 좌표 계산
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    box_radius = min(x2 - x1, y2 - y1) // 2

    # 크롭 (패딩 포함)
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

    # --- 2. 전처리: CLAHE + 적응형 이진화 (문제 해결 핵심) ---
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    _, _, v_ch = cv2.split(hsv)
    
    # CLAHE (명암비 향상)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v_ch)
    
    imwrite_korean(os.path.join(save_dirs['1_clahe'], filename), v_clahe)

    # [해결책] 적응형 이진화 (Adaptive Threshold)
    # 주변 픽셀(blockSize=21)보다 5(C)만큼 더 어두우면 확실하게 잡아냄
    # THRESH_BINARY_INV: 톱니(어두움)를 흰색(255)으로 반전
    binary = cv2.adaptiveThreshold(
        v_clahe, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        25, 5
    )

    # --- 3. 도넛 마스킹 (내부/외부 노이즈 제거) ---
    mask_donut = np.zeros_like(binary)
    r_out = int(box_radius * OUTER_RATIO)
    r_in = int(box_radius * INNER_RATIO)
    
    # 바깥 원은 255, 안쪽 원은 0으로 파내기 -> 도넛 모양만 255
    cv2.circle(mask_donut, center_crop, r_out, 255, -1)
    cv2.circle(mask_donut, center_crop, r_in, 0, -1)
    
    # 마스크 적용
    binary_masked = cv2.bitwise_and(binary, binary, mask=mask_donut)

    # 모폴로지 (노이즈 제거 및 덩어리 연결)
    kernel = np.ones((5, 5), np.uint8)
    # 톱니 내부의 구멍 메우기 (Close) 후 자잘한 점 제거 (Open)
    binary_clean = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    imwrite_korean(os.path.join(save_dirs['2_binary'], filename), binary_clean)

    # --- 4. 외곽선 추출 ---
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = cropped.copy()
    teeth_count = 0

    if contours:
        # 가장 큰 덩어리(기어 띠) 선택
        main_contour = max(contours, key=cv2.contourArea)
        
        # Convex Hull (고무줄)
        hull_indices = cv2.convexHull(main_contour, returnPoints=False)
        hull_points = cv2.convexHull(main_contour, returnPoints=True)

        # --- 5. [핵심] Convexity Defects (볼록 결함) 분석 ---
        # 톱니가 사다리꼴이어도 톱니 사이의 '골짜기'는 하나입니다.
        try:
            defects = cv2.convexityDefects(main_contour, hull_indices)
        except:
            defects = None

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                
                start = tuple(main_contour[s][0])
                end = tuple(main_contour[e][0])
                far = tuple(main_contour[f][0]) # 골짜기 가장 깊은 점
                
                # 깊이(d) 필터링: 너무 얕은 요철은 무시
                # d는 픽셀 거리 * 256 값이므로 나눠줌
                depth = d / 256.0
                
                if depth > 10: 
                    # 삼각형의 세 변 길이 계산 (Cosine 법칙용)
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    # 코사인 법칙으로 골짜기의 각도 계산
                    if b > 0 and c > 0: # 0으로 나누기 방지
                        angle_val = (b**2 + c**2 - a**2) / (2*b*c)
                        # 부동소수점 오차 보정 (-1 ~ 1 사이로 강제)
                        angle_val = max(-1.0, min(1.0, angle_val))
                        
                        angle_deg = math.degrees(math.acos(angle_val))
                        
                        # [조건] 골짜기의 각도가 90도 미만(뾰족함)이어야 톱니 사이로 인정
                        if angle_deg < 90:
                            teeth_count += 1
                            # 시각화: 골짜기 위치에 빨간 점
                            cv2.circle(result_img, far, 6, [0, 0, 255], -1)

        # 시각화: 외곽선(보라), Hull(초록)
        cv2.drawContours(result_img, [main_contour], -1, (255, 0, 255), 2)
        # cv2.drawContours(result_img, [hull_points], -1, (0, 255, 0), 1)

    # 텍스트 표시
    cv2.putText(result_img, f"Count: {teeth_count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    imwrite_korean(os.path.join(save_dirs['3_final_result'], filename), result_img)
    
    return teeth_count

# ==========================================
# 4. 실행 및 정확도 분석
# ==========================================
if __name__ == '__main__':
    print(f"🚀 [Convexity Defects Analysis] 시작")
    os.makedirs(output_root_folder, exist_ok=True)
    
    # 단계별 폴더 생성
    step_folders = ['0_crop', '1_clahe', '2_binary', '3_final_result']
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
            
        # 하위 폴더 생성
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
            
            # 예측 수행
            pred_count = process_gear_defects(img, file, current_save_dirs)
            
            if pred_count is not None:
                # 정답 파싱 (파일명의 첫 번째 부분) 예: 27_30.bmp -> 27
                try:
                    gt_str = file.split('_')[0]
                    gt_count = int(gt_str)
                    is_correct = (pred_count == gt_count)
                except:
                    gt_count = -1 # 파싱 실패
                    is_correct = False
                
                print(f"  - {file}: GT={gt_count}, Pred={pred_count} -> {'O' if is_correct else 'X'}")
                
                results_list.append({
                    'Folder': rel_path,
                    'Filename': file,
                    'Ground_Truth': gt_count,
                    'Predicted': pred_count,
                    'Correct': is_correct
                })

    # --- 정확도 리포트 생성 ---
    if results_list:
        df = pd.DataFrame(results_list)
        
        # 1. 파일별 결과 CSV 저장
        df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
        
        # 2. 통계 계산 (-1 제외)
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
            
            # 리포트 저장
            report_save_path = os.path.join(output_root_folder, 'accuracy_summary.csv')
            accuracy_report.to_csv(report_save_path, index=False, encoding='utf-8-sig')
            print(f"📄 리포트 저장 완료: {report_save_path}")

    print("\n✅ 모든 작업 완료. 17_convexity_defects_analysis 폴더를 확인하세요.")