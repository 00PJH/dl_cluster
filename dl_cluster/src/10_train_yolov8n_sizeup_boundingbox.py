from ultralytics import YOLO
from roboflow import Roboflow
import multiprocessing
import os

# ==========================================
# 1. 환경 설정 및 데이터셋 다운로드
# ==========================================
if __name__ == '__main__':
    # 윈도우 멀티프로세싱 에러 방지
    multiprocessing.freeze_support()
    
    # [설정] 결과가 저장될 경로를 절대 경로로 지정
    # project: 저장될 상위 폴더 (src 폴더)
    # name: 그 안에 생성될 폴더 이름 (train_yolo8n_sizeup_boundingbox)
    save_project_path = r"C:\Users\박준형\Desktop\python-workspace\dl_cluster\src"
    save_dir_name = "train_yolo8n_sizeup_boundingbox"

    print(f"🚀 학습 결과는 다음 경로에 저장됩니다: {os.path.join(save_project_path, save_dir_name)}")

    # Roboflow 데이터셋 다운로드 (Version 3)
    rf = Roboflow(api_key="C0fL7LVdzWSBeBsAqQla")
    project = rf.workspace("clusteralab").project("gear_detection")
    version = project.version(3) # v3 사용 (박스 크기 조절 버전)
    dataset = version.download("yolov8")
                
    # ==========================================
    # 2. YOLOv8 학습 시작
    # ==========================================
    
    # 모델 로드 (Nano 버전)
    model = YOLO('yolov8n.pt')

    # 학습 실행
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,       # 2060 Super 최적값
        patience=20,    # 조기 종료
        device=0,       # GPU 사용
        workers=4,      
        
        # [핵심 변경 사항] 저장 경로 분리 설정
        project=save_project_path,  # 결과를 저장할 상위 루트 폴더
        name=save_dir_name,         # 결과 폴더의 이름
        exist_ok=True               # 폴더가 있어도 에러내지 않고(덮어쓰거나 내부에 저장) 진행
    )

    print("\n✅ 학습이 완료되었습니다.")
    print(f"결과 파일(best.pt 등)을 확인하세요: {os.path.join(save_project_path, save_dir_name)}")