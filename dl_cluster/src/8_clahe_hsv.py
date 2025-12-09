import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수
# ==========================================
# 입력 데이터 경로 (사용자 환경에 맞게 수정)
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
# 결과 저장 경로
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\clahe_hsv'

# 톱니(산)로 인정할 최소 간격 (너무 자잘한 점 제거용)
MIN_PEAK_DISTANCE = 10 

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def imread_korean(file_path):
    """한글 경로 이미지 읽기"""
    try:
        img_array = np.fromfile(file_path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_korean(filename, img, params=None):
    """한글 경로 이미지 저장"""
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
# 3. 핵심 로직: CLAHE + 동적 임계값 + Peak 추출
# ==========================================
def extract_gear_peaks(img, filename):
    # 1. HSV 변환 및 채널 분리
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 2. [핵심] CLAHE 적용 (제공해주신 코드의 핵심 로직)
    # 금속 질감의 명암비를 극대화하여 톱니 경계를 뚜렷하게 만듦
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    
    # 3. 동적 임계값 적용 (Dynamic Thresholding)
    # 기존 코드의 '녹색' 대신, '명암'을 기준으로 배경과 물체를 자동 분리
    # Otsu 알고리즘: 히스토그램을 분석해 최적의 경계값을 자동으로 찾음 (완전 동적)
    # 배경이 밝고 기어가 어두우면 THRESH_BINARY_INV, 반대면 THRESH_BINARY
    # 보통 기어 사진은 배경이 밝으므로 INV를 시도합니다.
    _, mask = cv2.threshold(v_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. 형태학적 연산 (Morphology)
    # 끊어진 선을 잇고 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. 외곽선(Contour) 추출
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  ❌ 컨투어 검출 실패: {filename}")
        return img, 0

    # 가장 큰 덩어리가 톱니바퀴
    main_contour = max(contours, key=cv2.contourArea)
    
    # 6. [핵심] Convex Hull을 이용한 산(Peak) 추출
    # Convex Hull은 물체를 감싸는 고무줄 같은 다각형입니다.
    # 이 다각형의 꼭지점들이 바로 '톱니바퀴의 산(튀어나온 부분)'입니다.
    hull = cv2.convexHull(main_contour, returnPoints=True)
    
    # hull 포인트 형식 변환 (N, 1, 2) -> (N, 2)
    hull_points = hull.squeeze()
    
    # 피크 필터링 (너무 가까운 점들은 하나로 합침)
    final_peaks = []
    if len(hull_points) > 0:
        # y축 기준으로 정렬하거나, 그냥 순서대로 처리
        # 보통 Hull은 순서대로 나옵니다.
        final_peaks.append(hull_points[0])
        
        for i in range(1, len(hull_points)):
            pt = hull_points[i]
            prev_pt = final_peaks[-1]
            
            # 이전 점과의 거리 계산
            dist = np.linalg.norm(pt - prev_pt)
            
            # 일정 거리 이상 떨어진 점만 새로운 산으로 인정
            if dist > MIN_PEAK_DISTANCE:
                final_peaks.append(pt)
    
    # 7. 시각화
    result_img = img.copy()
    
    # (A) 찾은 외곽선 그리기 (보라색)
    cv2.drawContours(result_img, [main_contour], -1, (255, 0, 255), 2)
    
    # (B) Convex Hull(산의 경계) 그리기 (초록색)
    # 다시 그리기 위해 형태 변환
    hull_draw = np.array(final_peaks).reshape((-1, 1, 2))
    cv2.drawContours(result_img, [hull_draw], -1, (0, 255, 0), 2)
    
    # (C) 산(Peak) 꼭지점 찍기 (빨간점)
    for pt in final_peaks:
        cv2.circle(result_img, tuple(pt), 6, (0, 0, 255), -1)
        
    count = len(final_peaks)
    
    # 결과 텍스트
    cv2.putText(result_img, f"Peaks: {count}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    print(f"  - {filename} -> 산(Peak) 검출 개수: {count}")
    
    return result_img, count

# ==========================================
# 4. 실행 로직 (폴더 구조 유지)
# ==========================================
def run_process(root_folder):
    print("🚀 [최종] CLAHE 기반 톱니바퀴 산(Peak) 추출 시작")
    print(f"   소스 경로: {root_folder}")
    print(f"   저장 경로: {output_root_folder}")
    
    for root, dirs, files in os.walk(root_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue

        # 저장 폴더 구조 생성
        try:
            relative_path = os.path.relpath(root, input_root_folder)
        except:
            relative_path = os.path.basename(root)
            
        save_path = os.path.join(output_root_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n📂 처리 중인 폴더: {relative_path}")

        for file in bmp_files:
            file_path = os.path.join(root, file)
            img = imread_korean(file_path)
            if img is None: continue
            
            # 분석 실행
            result_img, count = extract_gear_peaks(img, file)
            
            # 저장
            save_file_path = os.path.join(save_path, file)
            imwrite_korean(save_file_path, result_img)

if __name__ == "__main__":
    run_process(input_root_folder)