from ultralytics import YOLO
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

quantmodel = YOLO("newyolo_def.yaml", "detect")

quantmodel.load("runs/detect/train92/weights/best.pt")

quantmodel.train(data="coco8.yaml", epochs=1, lr0=0.00001,momentum=0.85, save=True, device="cpu", batch=16,)

out_file = "runs/detect/train92/weights/quantized_yolo.onnx"
export_qonnx(quantmodel, export_path=out_file, args=torch.rand((1, 3, 320, 320), device=device))
qonnx_cleanup(out_file, out_file=out_file)

