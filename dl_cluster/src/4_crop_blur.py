# import cv2
# import numpy as np
# import os

# # ==========================================
# # 1. 설정 변수
# # ==========================================
# input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
# output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\cropped_blur'

# # 최종 이미지 크기
# target_size = 512 

# # [중요] 선명하게 남길 반지름 (중심 ~ 기와집 부분까지)
# # 512x512 이미지 기준, 이 거리 밖은 흐리게 처리됨
# clear_radius = 200 

# # ==========================================
# # 2. 유틸리티 함수
# # ==========================================
# def imread_korean(file_path):
#     try:
#         img_array = np.fromfile(file_path, np.uint8)
#         return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     except Exception:
#         return None

# def imwrite_korean(filename, img, params=None):
#     try:
#         ext = os.path.splitext(filename)[1]
#         result, n = cv2.imencode(ext, img, params)
#         if result:
#             with open(filename, mode='w+b') as f:
#                 n.tofile(f)
#             return True
#         return False
#     except Exception:
#         return False

# # ==========================================
# # 3. 핵심 로직
# # ==========================================
# def process_gear_image(img, filename):
#     h, w = img.shape[:2]
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # 1. 중심 찾기 (허프 변환)
#     blurred_gray = cv2.GaussianBlur(gray, (9, 9), 2)
#     circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w/2,
#                                param1=100, param2=30, minRadius=50, maxRadius=400)
    
#     cx, cy = w // 2, h // 2  # 기본값

#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         largest_circle = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[0]
#         # [수정 포인트] uint16 타입을 int로 변환하여 음수 계산 오류 방지!
#         cx, cy, r = int(largest_circle[0]), int(largest_circle[1]), int(largest_circle[2])
#         print(f"  - {filename} -> 중심: ({cx}, {cy}), 반지름: {r}")
#     else:
#         # 허프 변환 실패 시 무게 중심 사용
#         _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             c = max(contours, key=cv2.contourArea)
#             M = cv2.moments(c)
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 print(f"  - {filename} -> 무게 중심 사용: ({cx}, {cy})")

#     # 2. 512x512 크롭 (Padding 포함)
#     half_size = target_size // 2
#     canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8) # 검은 배경
    
#     # 원본에서의 좌표 (음수가 나올 수 있으므로 int 처리 필수)
#     src_x1 = cx - half_size
#     src_y1 = cy - half_size
#     src_x2 = cx + half_size
#     src_y2 = cy + half_size
    
#     # 실제 복사할 원본 범위 (이미지 밖으로 나가지 않게 클램핑)
#     img_x1 = max(0, src_x1)
#     img_y1 = max(0, src_y1)
#     img_x2 = min(w, src_x2)
#     img_y2 = min(h, src_y2)
    
#     # 캔버스에 붙여넣을 위치 계산
#     dst_x1 = max(0, img_x1 - src_x1)
#     dst_y1 = max(0, img_y1 - src_y1)
#     dst_x2 = dst_x1 + (img_x2 - img_x1)
#     dst_y2 = dst_y1 + (img_y2 - img_y1)
    
#     # 이미지 복사 (유효한 범위만)
#     if (img_x2 > img_x1) and (img_y2 > img_y1):
#         canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[img_y1:img_y2, img_x1:img_x2]
    
#     # 3. 블러 처리 (Blurring)
#     # 중심(256, 256) 기준 마스크 생성
#     mask = np.zeros((target_size, target_size), dtype=np.uint8)
#     center_crop = (target_size // 2, target_size // 2)
    
#     cv2.circle(mask, center_crop, clear_radius, 255, -1) # 안쪽은 유지(255)
    
#     # 전체 블러 이미지 생성
#     blurred_canvas = cv2.GaussianBlur(canvas, (21, 21), 0)
    
#     # 합성: 마스크가 있는 곳은 원본, 없는 곳은 블러
#     final_img = np.where(mask[..., None] > 0, canvas, blurred_canvas)

#     return final_img

# # ==========================================
# # 4. 실행 로직
# # ==========================================
# def run_process(root_folder):
#     print(f"🚀 [Fix 완료] 센터링 + 크롭 + 블러 처리 시작...")
    
#     for root, dirs, files in os.walk(root_folder):
#         bmp_files = [f for f in files if f.lower().endswith('.bmp')]
#         if not bmp_files: continue

#         folder_name = os.path.basename(root)
#         try:
#             relative_path = os.path.relpath(root, input_root_folder)
#         except:
#             relative_path = folder_name
            
#         save_path = os.path.join(output_root_folder, relative_path)
#         os.makedirs(save_path, exist_ok=True)
        
#         print(f"\n📂 처리 중: {relative_path}")

#         for file in bmp_files:
#             file_path = os.path.join(root, file)
#             img = imread_korean(file_path)
#             if img is None: continue
            
#             try:
#                 processed_img = process_gear_image(img, file)
#                 save_file_path = os.path.join(save_path, file)
#                 imwrite_korean(save_file_path, processed_img)
#             except Exception as e:
#                 print(f"  ❌ 에러 발생 ({file}): {e}")

# if __name__ == "__main__":
#     run_process(input_root_folder)

import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 변수
# ==========================================
input_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\data'
output_root_folder = r'C:\Users\박준형\Desktop\python-workspace\dl_cluster\cropped_blur_v3'

# [해결책 B] 톱니가 잘리지 않도록 넉넉하게 자르는 크기
source_crop_size = 1000  # 원본에서 1000px 만큼 뜯어냄 (톱니바퀴 크기에 맞춰 조절)

# 최종 결과물 크기
final_output_size = 512

# 블러 처리 기준 (512 크기 기준)
clear_radius = 230 

# ==========================================
# 2. 유틸리티 함수
# ==========================================
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
# 3. [핵심] 개선된 중심 찾기 알고리즘
# ==========================================
def find_center_robust(img, filename):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. 이진화 (Adaptive Threshold 대신 단순 Threshold 사용 권장 - 센터홀은 항상 어둡기 때문)
    # 주변보다 확실히 어두운 구멍을 찾습니다. (값 60은 조절 가능)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 외곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt = None
    min_dist_from_center = float('inf')
    
    img_cx, img_cy = w // 2, h // 2
    
    # 3. [해결책 A] 조건에 맞는 "진짜 구멍" 필터링
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # (1) 면적 필터: 너무 작거나(먼지) 너무 큰(배경) 것 제외
        if area < 500 or area > 50000:
            continue
            
        # (2) 원형도(Circularity) 필터: 찌그러진 그림자 제외
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 원형도가 0.7 이상인 것만 '원'으로 인정 (1.0이 완벽한 원)
        if circularity < 0.6: 
            continue
            
        # (3) 거리 필터: 이미지 물리적 중앙에서 너무 먼 것은 오검출일 확률 높음
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        
        # 가장 중앙에 가깝고, 조건에 맞는 녀석을 선택
        if dist < min_dist_from_center:
            min_dist_from_center = dist
            best_cnt = (cx, cy)

    if best_cnt:
        print(f"  - {filename} -> 정밀 중심 검출 성공: {best_cnt}")
        return best_cnt
    else:
        # 실패 시 그냥 이미지 정중앙 리턴 (에러 방지)
        print(f"  - {filename} -> 검출 실패 (중앙값 사용)")
        return (img_cx, img_cy)

# ==========================================
# 4. 전체 처리 로직
# ==========================================
def process_gear_image(img, filename):
    h, w = img.shape[:2]
    
    # 1. 개선된 중심 찾기 실행
    cx, cy = find_center_robust(img, filename)
    
    # 2. [해결책 B] 넉넉하게 크롭 (Source Crop)
    half_source = source_crop_size // 2
    crop_canvas = np.zeros((source_crop_size, source_crop_size, 3), dtype=np.uint8)
    
    src_x1 = int(cx - half_source)
    src_y1 = int(cy - half_source)
    src_x2 = int(cx + half_source)
    src_y2 = int(cy + half_source)
    
    # 원본 범위 클램핑
    img_x1 = max(0, src_x1)
    img_y1 = max(0, src_y1)
    img_x2 = min(w, src_x2)
    img_y2 = min(h, src_y2)
    
    # 붙여넣을 위치 계산
    dst_x1 = max(0, img_x1 - src_x1)
    dst_y1 = max(0, img_y1 - src_y1)
    dst_x2 = dst_x1 + (img_x2 - img_x1)
    dst_y2 = dst_y1 + (img_y2 - img_y1)
    
    if (img_x2 > img_x1) and (img_y2 > img_y1):
        crop_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[img_y1:img_y2, img_x1:img_x2]

    # 3. 리사이즈 (512x512)
    final_img = cv2.resize(crop_canvas, (final_output_size, final_output_size), interpolation=cv2.INTER_AREA)

    # 4. 블러 처리
    mask = np.zeros((final_output_size, final_output_size), dtype=np.uint8)
    center_final = (final_output_size // 2, final_output_size // 2)
    cv2.circle(mask, center_final, clear_radius, 255, -1)
    
    blurred_bg = cv2.GaussianBlur(final_img, (21, 21), 0)
    result = np.where(mask[..., None] > 0, final_img, blurred_bg)

    return result

def run_process(root_folder):
    print(f"🚀 [V3] 위치 보정 및 와이드 크롭 프로세스 시작...")
    
    for root, dirs, files in os.walk(root_folder):
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        if not bmp_files: continue

        folder_name = os.path.basename(root)
        try:
            relative_path = os.path.relpath(root, input_root_folder)
        except:
            relative_path = folder_name
            
        save_path = os.path.join(output_root_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n📂 처리 중: {relative_path}")

        for file in bmp_files:
            file_path = os.path.join(root, file)
            img = imread_korean(file_path)
            if img is None: continue
            
            try:
                processed_img = process_gear_image(img, file)
                save_file_path = os.path.join(save_path, file)
                imwrite_korean(save_file_path, processed_img)
            except Exception as e:
                print(f"  ❌ 에러 발생 ({file}): {e}")

if __name__ == "__main__":
    run_process(input_root_folder)