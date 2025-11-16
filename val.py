import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/ps/ultralytics-202404066/ultralytics-main/runs/train/exp_yolov8n-FDPN-LSCD1_nwpu/weights/best.pt')
    model.val(data='/home/ps/ultralytics-202404066/ultralytics-main/ultralytics/cfg/datasets/nwpu.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp_yolov8n-FDPN-LSCD1_nwpu111',
              )



