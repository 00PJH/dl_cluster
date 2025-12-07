import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수
# ==========================================
# 요청하신 입력 데이터 경로 (한글 경로 포함)
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'

# 결과를 저장할 경로 (data 폴더와 같은 레벨의 final 폴더에 저장)
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\final'

# 검사할 도넛 모양 영역의 반지름 설정 (이미지 크기에 맞춰 조절 필요)
radius_inner = 130  # 안쪽 원 반지름
radius_outer = 630  # 바깥쪽 원 반지름

# 결함(또는 특징)으로 인식할 면적 기준
min_area = 30     # 너무 작은 점(노이즈) 제거
max_area = 5000    # 너무 큰 영역 제외

# ==========================================
# 2. 한글 경로 처리 함수 (필수!)
# ==========================================
def imread_korean(file_path):
    """한글 경로가 포함된 이미지를 읽어오는 함수"""
    try:
        # 파일을 바이트 단위로 읽어서 디코딩
        img_array = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"이미지 읽기 실패: {e}")
        return None

def imwrite_korean(filename, img, params=None):
    """한글 경로에 이미지를 저장하는 함수"""
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
        return False

# ==========================================
# 3. 핵심 로직 함수
# ==========================================
def find_center_of_object(img_gray):
    """제품의 중심 좌표(cx, cy)를 자동으로 찾습니다."""
    # 배경과 물체를 분리 (Threshold 값은 상황에 따라 50~127 조절)
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 가장 큰 외곽선이 제품일 확률이 높음
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def process_images(root_folder):
    print(f"🚀 작업을 시작합니다... (경로: {root_folder})")
    
    # os.walk를 사용하여 하위 폴더까지 모두 탐색
    for root, dirs, files in os.walk(root_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        
        if not bmp_files:
            continue

        # 현재 처리 중인 하위 폴더 이름 (예: 27_30)
        folder_name = os.path.basename(root)
        
        # 상대 경로 계산 (data 폴더 내부의 구조를 그대로 따르기 위함)
        # 예: data/27_30 -> relative_path는 "27_30"
        try:
            relative_path = os.path.relpath(root, input_root_folder)
        except ValueError:
            relative_path = folder_name

        print(f"\n📂 폴더 처리 중: {relative_path}")

        # 저장할 폴더 생성 (final 폴더 내부에 동일한 구조 생성)
        save_path = os.path.join(output_root_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)

        for file in bmp_files:
            file_path = os.path.join(root, file)
            
            # [수정됨] 한글 경로 처리를 위해 커스텀 함수 사용
            img = imread_korean(file_path)
            
            if img is None:
                print(f"  ❌ 파일 로드 실패: {file}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. 중심 찾기
            center = find_center_of_object(gray)
            if center is None:
                h, w = gray.shape
                center = (w // 2, h // 2) # 못 찾으면 중앙 사용
            
            cx, cy = center

            # 2. ROI 마스크 생성 (도넛)
            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), radius_outer, 255, -1)
            cv2.circle(mask, (cx, cy), radius_inner, 0, -1)
            
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            
            # 3. 특징 검출
            # 밝은 링 위의 어두운 특징을 찾으므로 THRESH_BINARY_INV 사용
            # (만약 특징이 잘 안 잡히면 80 값을 100 정도로 올려보세요)
            _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_and(binary, binary, mask=mask)

            # 4. 개수 세기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            total_area = 0.0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    valid_contours.append(cnt)
                    total_area += area
            
            count = len(valid_contours)
            avg_area = total_area / count if count > 0 else 0.0

            # 5. 결과 그리기 및 저장
            result_img = img.copy()
            cv2.circle(result_img, (cx, cy), radius_inner, (0, 255, 255), 2)
            cv2.circle(result_img, (cx, cy), radius_outer, (0, 255, 255), 2)
            cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 2)
            
            text = f"Count: {count}  Area: {avg_area:.1f}"
            cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            save_file_path = os.path.join(save_path, file)
            
            # [수정됨] 한글 경로 저장을 위해 커스텀 함수 사용
            if imwrite_korean(save_file_path, result_img):
                print(f"  - {file} -> 개수: {count}, 면적: {avg_area:.1f}")
            else:
                print(f"  ❌ 저장 실패: {file}")

if __name__ == "__main__":
    process_images(input_root_folder)