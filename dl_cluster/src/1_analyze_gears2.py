import cv2
import numpy as np
import os
import glob

# ==========================================
# 0. 한글 경로 지원을 위한 헬퍼 함수 (추가됨)
# ==========================================
def imread_korean(path):
    """한글 경로가 포함된 이미지를 읽어오는 함수"""
    try:
        # 파일을 바이너리로 읽어서 numpy 배열로 변환 후 디코딩
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[Error] 이미지 읽기 실패: {path} \n{e}")
        return None

def imwrite_korean(path, img):
    """한글 경로에 이미지를 저장하는 함수"""
    try:
        # 이미지 확장자 추출 (.jpg, .bmp 등)
        extension = os.path.splitext(path)[1]
        result, encoded_img = cv2.imencode(extension, img)
        if result:
            with open(path, mode='w+b') as f:
                encoded_img.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(f"[Error] 이미지 저장 실패: {path} \n{e}")
        return False

# ==========================================
# 1. 환경 설정 및 파라미터
# ==========================================

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src 폴더
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # dl_cluster 폴더
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')             # 데이터 폴더
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')       # 결과 저장 폴더

# 톱니바퀴 인식 파라미터
MIN_TOOTH_AREA = 50     
MAX_TOOTH_AREA = 1000   
MIN_DIST_FROM_CENTER = 100 

# [튜닝 포인트] 30도 vs 45도 분류 기준값 (면적 기반)
# 실행 결과를 보고 이 값을 조정하세요.
ANGLE_CLASSIFY_THRESHOLD = 400 

# ==========================================
# 2. 핵심 알고리즘 함수
# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[Info] 폴더 생성: {directory}")

def get_image_center(img_shape):
    h, w = img_shape[:2]
    return (w // 2, h // 2)

def analyze_gear(image_path, filename):
    # 수정됨: cv2.imread 대신 imread_korean 사용
    img = imread_korean(image_path)
    
    if img is None:
        return None, None, 0, 0
    
    # 1. 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 외곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    center_x, center_y = get_image_center(img.shape)
    gear_teeth_contours = []
    total_area = 0

    # 3. 톱니 필터링
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        dist_from_center = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)

        if MIN_TOOTH_AREA < area < MAX_TOOTH_AREA and dist_from_center > MIN_DIST_FROM_CENTER:
            gear_teeth_contours.append(cnt)
            total_area += area

    count = len(gear_teeth_contours)
    avg_area = total_area / count if count > 0 else 0

    # 시각화
    result_img = img.copy()
    cv2.drawContours(result_img, gear_teeth_contours, -1, (0, 255, 0), 2)
    
    cv2.putText(result_img, f"Count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(result_img, f"Avg Area: {avg_area:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return result_img, gear_teeth_contours, count, avg_area

# ==========================================
# 3. 메인 실행 로직
# ==========================================

def main():
    print("=== 기어 분석 시작 (한글 경로 패치 버전) ===")
    
    # 데이터 폴더가 실제로 있는지 확인
    if not os.path.exists(DATA_DIR):
        print(f"[Error] 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        return

    subfolders = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]
    
    contour_save_dir = os.path.join(RESULTS_DIR, 'contour')
    angle_30_save_dir = os.path.join(RESULTS_DIR, 'angle_30')
    angle_45_save_dir = os.path.join(RESULTS_DIR, 'angle_45')
    
    ensure_dir(contour_save_dir)
    ensure_dir(angle_30_save_dir)
    ensure_dir(angle_45_save_dir)

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"\n📂 처리 중인 폴더: {folder_name}")
        
        image_files = glob.glob(os.path.join(folder, "*.bmp"))
        
        if not image_files:
            print("  -> 이미지 파일(.bmp)이 없습니다.")
            continue

        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            # 분석 수행
            res_img, contours, count, avg_area = analyze_gear(img_path, filename)
            
            if res_img is None:
                continue

            print(f"  - {filename}: 개수={count}, 평균면적={avg_area:.1f}")

            # -------------------------------------------------
            # 저장 로직 (imwrite_korean 사용)
            # -------------------------------------------------
            save_name = f"{os.path.splitext(filename)[0]}_cnt{count}.jpg"
            
            # 1. Contour 저장
            imwrite_korean(os.path.join(contour_save_dir, save_name), res_img)

            # 2. 각도 분류 저장 (27개인 경우만)
            if count == 27:
                if avg_area > ANGLE_CLASSIFY_THRESHOLD: 
                    prediction = "45"
                    save_target_dir = angle_45_save_dir
                else:
                    prediction = "30"
                    save_target_dir = angle_30_save_dir
                
                final_img = res_img.copy()
                cv2.putText(final_img, f"Pred: {prediction} deg", (10, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                imwrite_korean(os.path.join(save_target_dir, save_name), final_img)

    print("\n=== 모든 처리 완료 ===")
    print(f"결과 확인 경로: {RESULTS_DIR}")

if __name__ == "__main__":
    main()