import os
import torch
import cv2
import argparse
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', action='store_true')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--model_extension', type=str, default='engine', help='pt or engine (engine is much faster)')
    parser.add_argument('--model_size', type=str, default='m')
    parser.add_argument('--resolution', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.6)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    data_root = args.data_root # this should be one file if single_frame is True
    model_extension = args.model_extension
    resolution = args.resolution
    conf = args.conf
    device = args.device

    model_name = f'{args.model_size}_{resolution}'
    model_file = f'runs/pose/{model_name}/weights/best.{model_extension}'

    model = YOLO(model_file)
    if model_extension == 'pt' and device != 'cpu' and torch.cuda.is_available():
        model = model.to(device)


    if args.sequence:
        frames = [int(file.split('.')[0]) for file in os.listdir(data_root) if file.endswith('.png')]
        start_frame, end_frame = min(frames), max(frames)
        for frame_num in range(start_frame, end_frame + 1):
            frame_file = f'{frame_num}.png'
            frame_path = os.path.join(data_root, frame_file)
            results = model.track(frame_path, conf=conf, imgsz=resolution, device=device, persist=True, tracker='bytetrack.yaml')
            img = results[0].plot()
            cv2.imshow('result', img)
            cv2.waitKey(500)

    else:
        results = model(data_root, conf=conf, imgsz=resolution, device=device)
        for result in results:
            img = result.plot()
            cv2.imshow('result', img)
            cv2.waitKey(500)
