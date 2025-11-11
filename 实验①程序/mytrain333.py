import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # ← 禁用 OpenMP 多线程
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    model.train(
        data=r'D:\YOLO\SeaShips_YOLO\data.yaml',
        epochs=100,
        imgsz=640,
        batch=-1,
        project='SeaShips_SMD',
        name='mini_train',
        device=0,
        workers=6,   # ← 建议先设为 2（见第二步）
        exist_ok=True
    )

if __name__ == '__main__':
    main()