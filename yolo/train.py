import torch
from ultralytics import YOLO
torch.cuda.empty_cache()

# Load a model
res = 512
model_size = 's'
model = YOLO(f'yolov8{model_size}-pose.pt')  # load a pretrained model (recommended for training)

# Train the model (this might take a day on our synthetic dataset depending on the GPU)
results = model.train(data='surgical.yaml', epochs=200, imgsz=res, batch=8, plots=True, workers=1, name=f'{model_size}_{res}', exist_ok=True, fraction=1.0)

results = model.val(data='surgical.yaml', split='test')

model.export(format='engine')