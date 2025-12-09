from ultralytics import YOLO
from roboflow import Roboflow
import multiprocessing # 혹시 모를 에러 방지용

# ==========================================
# 1. 데이터셋 다운로드
# ==========================================
# (이 부분은 함수 밖이나 안에 있어도 되지만, 안전하게 main 안에 넣는 것을 추천합니다)

# ==========================================
# 2. YOLOv8 학습 시작 (메인 실행부)
# ==========================================
if __name__ == '__main__':
    # 멀티프로세싱 안전장치 (윈도우 필수)
    multiprocessing.freeze_support() 
    
    rf = Roboflow(api_key="C0fL7LVdzWSBeBsAqQla")
    project = rf.workspace("clusteralab").project("gear_detection")
    version = project.version(2)
    dataset = version.download("yolov8")
                
    # ----------------------------------------------

    # 모델 로드
    model = YOLO('yolov8n.pt')

    # 학습 실행
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        workers=4,  # 이 옵션 때문에 main 블록이 필수입니다
        name='gear_train_v1'
    )