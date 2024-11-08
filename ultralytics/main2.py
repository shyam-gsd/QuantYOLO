#!/home/sjoshi/QuantYOLO/venv/bin/python
import sys

sys.path.append('/home/sjoshi/QuantYOLO/ultralytics/ultralytics')
sys.path.append('/home/sjoshi/QuantYOLO/brevitas/src')



from ultralytics import YOLO


if __name__ == '__main__':


    Model = YOLO("newyolo_def.yaml", "detect")

    Model.train(data="coco.yaml", epochs=200, lr0=0.00001,momentum=0.85, save=True, device=[0,1], batch=16,
                     val=True, imgsz=320, optimizer="Adam",
                     project="Quantization")


