from brevitas.graph.calibrate import bias_correction_mode, calibration_mode
from torchvision import transforms, datasets

from tqdm import tqdm

from ultralytics import YOLO
import torch
import  json

def find_differences(dict1, dict2):
    # Initialize difference dictionaries
    diff = {'only_in_dict1': {}, 'only_in_dict2': {}, 'different_values': {}}

    # Compare keys and values
    for key in dict1:
        if key not in dict2:
            diff['only_in_dict1'][key] = key
        #Handle potential ambiguous tensor comparisons
        # elif isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
        #     if not torch.equal(dict1[key], dict2[key]):  # Use torch.equal to compare tensors
        #         diff['different_values'][key] = {'dict1': dict1[key], 'dict2': dict2[key]}
        # elif dict1[key] != dict2[key]:
        #     diff['different_values'][key] = {'dict1': dict1[key], 'dict2': dict2[key]}

    for key in dict2:
        if key not in dict1:
            diff['only_in_dict2'][key] = key

    return diff


# Build a YOLOv6n model from scratch
floatModel = YOLO("runs/detect/train92/weights/best.pt")


quantModel = YOLO("newyolo_def.yaml")
quantModel.load("runs/detect/train92/weights/best.pt")
# quantModel.load_state_dict(floatModel.state_dict())


diffs = find_differences(floatModel.state_dict(), quantModel.state_dict())
# print(json.dumps(diffs,indent=4))
#
print(f"Total values float : {len(floatModel.state_dict().keys())}")
print(f"Total values quant : {len(quantModel.state_dict().keys())}")
print(f"only in dict1 : {len(diffs['only_in_dict1'].keys())}")
print(f"only in dict2 : {len(diffs['only_in_dict2'].keys())}")
print(f"differences : {len(diffs['different_values'].keys())}")



# Display model information (optional)
quantModel.info()



# Train the model on the COCO8 example dataset for 100 epochs

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def get_dataloader(folder, size):
    # dataset = datasets.ImageFolder(folder, transforms.ToTensor())
    dataset = datasets.ImageFolder(folder, transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ]))
    return torch.utils.data.DataLoader(dataset)
dataloader = get_dataloader("ultralytics/images", 320)
with torch.no_grad():
    print("Calibrate:")
    with calibration_mode(quantModel):
        for x, _ in tqdm(dataloader):
            x = quantModel(x.to(device))

    print("Bias Correction:")
    with bias_correction_mode(quantModel), bias_correction_mode(quantModel):
        for x, _ in tqdm(dataloader):
            x = quantModel(x.to(device))


results = quantModel.train(data="data.yaml",  imgsz=320,  save=False)