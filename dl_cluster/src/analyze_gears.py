
import cv2
import numpy as np
import glob
import os
from scipy.signal import find_peaks, peak_widths

def analyze_gear(image_path):
    # 경로에서 파일명과 상위 폴더(정답 라벨) 추출
    filename = os.path.basename(image_path)
    label = os.path.basename(os.path.dirname(image_path))
    
    # 한글 경로 지원을 위한 이미지 로드 방식
    # cv2.imread는 한글 경로에서 실패할 수 있음 -> numpy로 읽어서 디코딩
    try:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return {'filename': filename, 'label': label, 'error': f'로드 실패: {e}'}

    if img is None:
        return {'filename': filename, 'label': label, 'error': '로드 실패 (None)'}
    
    h_img, w_img = img.shape[:2]
    img_area = h_img * w_img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) 
    
    # Otsu 이진화
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for c in contours:
        area = cv2.contourArea(c)
        # 노이즈 및 너무 큰 배경 제거 (1024x768 기준 미세 조정 가능)
        if area < img_area * 0.001: continue
        if area > img_area * 0.99: continue
            
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        
        # 중앙에 위치한 객체 우선
        dist_from_center = np.sqrt((cX - w_img/2)**2 + (cY - h_img/2)**2)
        if dist_from_center > min(w_img, h_img) * 0.4: 
            continue
            
        candidates.append({
            'contour': c,
            'area': area,
            'center': (cX, cY)
        })
        
    if not candidates:
        return {'filename': filename, 'label': label, 'error': '기어 윤곽선 없음'}
        
    candidates.sort(key=lambda x: x['area'], reverse=True)
    
    best_result = None
    
    # 상위 5개 후보에 대해 기어 신호 검사
    for cand in candidates[:5]:
        c = cand['contour']
        cX, cY = cand['center']
        
        # 1. 윤곽선 펼치기 (극좌표 변환)
        pts = c.reshape(-1, 2)
        rel_pts = pts - np.array([cX, cY])
        
        radii = np.linalg.norm(rel_pts, axis=1)
        thetas = np.arctan2(rel_pts[:, 1], rel_pts[:, 0])
        
        # 2. 각도 순 정렬 및 보간
        idx = np.argsort(thetas)
        thetas_sorted = thetas[idx]
        radii_sorted = radii[idx]
        
        if len(radii_sorted) < 50: continue

        target_thetas = np.linspace(-np.pi, np.pi, 2048) 
        radii_interp = np.interp(target_thetas, thetas_sorted, radii_sorted, period=2*np.pi)
        
        # 3. 추세 제거 (Detrend)
        window = 200
        radii_trend = np.convolve(radii_interp, np.ones(window)/window, mode='same')
        signal = radii_interp - radii_trend
        
        sig_range = np.max(signal) - np.min(signal)
        avg_radius = np.mean(radii_sorted)
        
        # 신호가 너무 매끄러우면 링(Ring)일 확률 높음 -> 제외
        if avg_radius > 0 and (sig_range / avg_radius) < 0.02:
            continue

        # 4. 피크 검출
        min_dist = 30 # 톱니 간 최소 거리
        prominence = sig_range * 0.15
        
        peaks, p_props = find_peaks(signal, prominence=prominence, distance=min_dist)
        valleys, v_props = find_peaks(-signal, prominence=prominence, distance=min_dist)
        
        p_count = len(peaks)
        v_count = len(valleys)
        
        final_count = 0
        metric_peaks = []
        metric_props = {}
        is_peaks = True
        
        # 유효 범위 (15 ~ 60개)
        valid_p = 15 <= p_count <= 60
        valid_v = 15 <= v_count <= 60
        
        if valid_p and not valid_v:
            final_count = p_count
            metric_peaks = peaks
            metric_props = p_props
        elif valid_v and not valid_p:
            final_count = v_count
            metric_peaks = valleys
            metric_props = v_props
            signal = -signal # 계곡을 피크로 반전
        elif valid_p and valid_v:
            # 둘 다 유효하면 28이나 31에 가까운 것 선택 (또는 분산이 큰 것)
            if abs(p_count - 28) < abs(v_count - 28):
                final_count = p_count
                metric_peaks = peaks
                metric_props = p_props
            else:
                final_count = v_count
                metric_peaks = valleys
                metric_props = v_props
                signal = -signal
        else:
            continue
            
        # 5. 각도 메트릭 계산 (뾰족함 정도)
        widths = []
        proms = []
        
        if len(metric_peaks) > 0:
            w_results = peak_widths(signal, metric_peaks, rel_height=0.5)
            widths = w_results[0]
            proms = metric_props['prominences']
            
        avg_width = np.mean(widths) if len(widths) > 0 else 0
        avg_prom = np.mean(proms) if len(proms) > 0 else 0
        
        angle_metric = 0
        if avg_width > 0:
            angle_metric = avg_prom / avg_width 
            
        best_result = {
            'filename': filename,
            'label': label,
            'count': final_count,
            'avg_width': avg_width, 
            'avg_prom': avg_prom,
            'angle_metric': angle_metric
        }
        break 
    
    if best_result:
        return best_result
    else:
        return {
            'filename': filename,
            'label': label,
            'error': '유효한 기어 패턴 없음'
        }

def main():
    # 데이터 경로 설정 (비트맵 파일 재귀 검색)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    image_files = glob.glob(os.path.join(data_dir, '**', '*.bmp'), recursive=True)
    
    results = []
    
    print(f"데이터 폴더 검색: {data_dir}")
    print(f"총 {len(image_files)}개의 비트맵(.bmp) 파일을 발견했습니다. 분석을 시작합니다...\n")
    
    for img_path in image_files:
        try:
            res = analyze_gear(img_path)
            results.append(res)
        except Exception as e:
            print(f"에러 {img_path}: {e}")
                
    # 라벨 및 파일명 정렬
    results.sort(key=lambda x: (x.get('label', ''), x.get('filename', '')))
    
    print("="*100)
    print(f"{'라벨':<10} | {'파일명':<30} | {'개수':<5} | {'뾰족함(Metric)':<15} | {'너비(Width)':<10}")
    print("="*100)
    
    correct_count = 0
    total_analyzed = 0

    for r in results:
        if 'error' in r:
            print(f"{r['label']:<10} | {r['filename']:<30} | {r['error']}")
        else:
            total_analyzed += 1
            print(f"{r['label']:<10} | {r['filename']:<30} | {r['count']:<5} | {r['angle_metric']:<15.4f} | {r['avg_width']:<10.4f}")
            
            # 간단한 검증 로직 (라벨에 숫자가 포함되어 있으면 비교)
            # 예: "28" -> 28개여야 함. "27_30" -> 27개여야 함.
            label_num = ''.join(filter(str.isdigit, r['label'].split('_')[0]))
            if label_num and int(label_num) == r['count']:
                correct_count += 1
                
    print("="*100)
    if total_analyzed > 0:
        print(f"\n[요약] 분석 완료: {total_analyzed}개 성공")
        # print(f"단순 개수 일치율 (참고용): {correct_count}/{total_analyzed} ({(correct_count/total_analyzed)*100:.1f}%)")
    else:
        print("\n분석된 파일이 없습니다.")

if __name__ == "__main__":
    main()
