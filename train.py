import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/CLADet/ultralytics/cfg/models/v8/yolov8n-CLAD-LFTA.yaml')
   
    model.train(data='/CLADet/ultralytics/cfg/datasets/DOTA-v1.0.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=200,
                workers=8,
                device='0',
                optimizer='SGD', 
                amp=False, 
                # fraction=0.2,
                project='runs/train',
                name='exp_yolov8n-CLAD-LFTA_DOTA-v1.0',
                )



