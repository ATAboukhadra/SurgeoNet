import torch
import cv2
import os
import numpy as np
import time

from ultralytics import YOLO
from tqdm import tqdm
from utils.preprocessor import Preprocessor
from transformer.train_utils import get_mesh
from utils.inference_utils import find_matches, Trackable, colors, visualize_meshes, init_outputs, fixer
from utils.params import ParamGroup
from utils.varjo_utils import read_ext
from utils.mesh_utils import load_objects

from utils.optimizer import PoseOptimizer, mpjpe

from transformer.dataset import one_hot_encode
from transformer.transformer import Transformer

import argparse

def predict_sequence(args):
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    conf = args.conf
    num_kps = 12

    # Load YOLO
    resolution = args.resolution
    yolo_name = f'{args.model_size}_{resolution}'
    yolo = {'left': YOLO(f'yolo/runs/pose/{yolo_name}/weights/best.{args.model_extension}'),
            'right': YOLO(f'yolo/runs/pose/{yolo_name}/weights/best.{args.model_extension}')}  # load a custom model
    if args.model_extension == 'pt' and device != 'cpu' and torch.cuda.is_available():
        yolo['left'] = yolo['left'].to(device)
        yolo['right'] = yolo['right'].to(device)

    # Load Transformer
    input_dim = 2 + 2 + 14 + num_kps # 2D keypoint + one-hot obj class
    output_dim = 6 + 3 + 4 # 6D rotation + 3 translation + quaternion articulation angle
    hidden_dim = 128
    num_layers = 5
    num_heads = 8
    use_transformer = args.use_transformer
    experiment_name = 'yolo_optimizer' if not use_transformer else 'yolo_transformer'

    if use_transformer:
        pose_estimator = Transformer(input_dim, hidden_dim, output_dim, num_layers, num_heads).to(device)
        pose_estimator.eval()
        pose_estimator.load_state_dict(torch.load(f'transformer/transformer.pth'))

    fix_kps = args.fix_kps
    reprojection_errors = []

    # Load data
    sequence_name = args.sequence_name
    root = os.path.join(args.data_root, sequence_name)
    if args.save: os.makedirs(f'output/{experiment_name}/{sequence_name}', exist_ok=True)

    # Load meta data
    dim = 1152
    fx, fy = 0.625, 0.625
    cx, cy = 0.5, 0.5
    cam_int = torch.tensor([[fx * dim, 0., cx * dim],
                    [0., fy * dim, cy * dim],
                    [0., 0., 1.]]).to(device)
    
    _, _, cam_ext_left = read_ext('varjo_calibration/LeftCameraExtrinsics.txt', device)
    _, _, cam_ext_right = read_ext('varjo_calibration/RightCameraExtrinsics.txt', device)
    cam_ext = [cam_ext_left, cam_ext_right]
    all_meshes, _ = load_objects(device, range(1, 14), nKps=num_kps)

    files = sorted(os.listdir(root))
    frames = [int(file.split('.')[0].split('_')[-1]) for file in files if file.endswith('.bin')]
    start_frame, end_frame = min(frames), max(frames)
    preprocessor = Preprocessor(device, root, dim)
    prev_pred_pose = torch.zeros((1, 10)).to(device)
    prev_pred_kps = torch.zeros((1, num_kps, 2)).to(device)
    found = False
    detections = {'left': {}, 'right': {}}
    
    reprojection_errors = []
    yolo_pose_fps_list = []

    for frame_num in tqdm(range(start_frame, end_frame+1)):
        image_pair = preprocessor.readPair(frame_num)
        if image_pair is None:
            continue

        if args.save:
            init_outputs(image_pair, experiment_name, sequence_name, frame_num)

        img1 = np.copy(image_pair['left'])
        img2 = np.copy(image_pair['right'])
        img_kf = [np.copy(img1), np.copy(img2)]

        imagePairList = [image_pair['left'], image_pair['right']]
        yoloPreds = {}
        t2 = time.time()

        yoloPreds = yolo['left'].track(imagePairList[0], conf=conf, persist=True, tracker='bytetrack.yaml', agnostic_nms=True, imgsz=resolution, verbose=False)\
            , yolo['right'].track(imagePairList[1], conf=conf, persist=True, tracker='bytetrack.yaml', agnostic_nms=True, imgsz=resolution, verbose=False)
            
        valid = True
        new_ids = {'left': [], 'right': []}
        yolo_plots = []

        for i, camera in enumerate(['left', 'right']):
            
            if len(yoloPreds[i]) == 0:
                continue
            
            yoloRes = yoloPreds[i][0]
            if yoloRes.boxes.shape[0] == 0 or yoloRes.boxes.id is None:
                continue

            yolo_plots.append(yoloRes.plot()[:, :, ::-1])

            for boxi in range(yoloRes.boxes.shape[0]):

                bbw = yoloRes.boxes.xyxy[boxi, 2] - yoloRes.boxes.xyxy[boxi, 0]
                bbh = yoloRes.boxes.xyxy[boxi, 3] - yoloRes.boxes.xyxy[boxi, 1]
                x_top_left, y_top_left = yoloRes.boxes.xyxyn[boxi, 0], yoloRes.boxes.xyxyn[boxi, 1]
                x_bottom_right, y_bottom_right = yoloRes.boxes.xyxyn[boxi, 2], yoloRes.boxes.xyxyn[boxi, 3]
                bbox = torch.tensor([[x_top_left, y_top_left], [x_bottom_right, y_bottom_right]], device=device)
                box_id = int(yoloRes.boxes.id[boxi])
                new_ids[camera].append(box_id)
                pred_label = int(yoloRes.boxes.cls[boxi])
                yoloPredsKps = yoloRes.keypoints.xyn.detach().clone()[boxi, :].unsqueeze(0)
                for j in range(yoloPredsKps.shape[1]):
                    x, y = yoloPredsKps[0, j, 0] * dim, yoloPredsKps[0, j, 1] * dim
                    img_kf[i] = cv2.circle(img_kf[i], (int(x), int(y)), 3, (255, 0, 0), -1)

                if box_id not in detections[camera].keys():
                    detections[camera][box_id] = Trackable(bbox, yoloPredsKps, pred_label, bb_size=max(bbw, bbh).item() / dim, 
                                                           conf=yoloRes.boxes.conf[boxi].item(), num_kps=num_kps, device=device)
                else:
                    detections[camera][box_id].update(bbox, yoloPredsKps, pred_label, bb_size=max(bbw, bbh).item() / dim, conf=yoloRes.boxes.conf[boxi].item())

        matches = find_matches(detections['left'], detections['right'], new_ids)

        if len(matches) == 0:
            continue
        sizes = []
        t3 = time.time()
        if use_transformer:
            transformerInput = torch.zeros(len(matches), num_kps, input_dim).to(device)
            # the element that occurs the most
            keypoint_classes = one_hot_encode(range(0, num_kps), num_classes=num_kps).to(device).unsqueeze(0).repeat(len(matches), 1, 1)
            transformerInput[:, :, -num_kps:] = keypoint_classes

            m_labels = []
            for sample_idx, (box_id_left, box_id_right, m_label) in enumerate(matches):
                m_labels.append(m_label)
                transformerInput[sample_idx, :, 4 + m_label] = 1

                kps_left = detections['left'][box_id_left].get_kps(detections['right'][box_id_right].kps, detections['right'][box_id_right].conf)
                kps_right = detections['right'][box_id_right].get_kps(kps_left, detections['left'][box_id_left].conf)

                if fix_kps:
                    kps_left_fixed, kps_right_fixed, is_fixed = fixer(detections['left'][box_id_left], detections['right'][box_id_right])

                    for j in range(yoloPredsKps.shape[1]):
                        x, y = kps_left[0, j, 0] * dim, kps_left[0, j, 1] * dim
                        x_fixed, y_fixed = kps_left_fixed[0, j, 0] * dim, kps_left_fixed[0, j, 1] * dim
                        img_kf[0] = cv2.circle(img_kf[0], (int(x), int(y)), 3, (0, 255, 0), -1)
                        img_kf[0] = cv2.circle(img_kf[0], (int(x_fixed), int(y_fixed)), 3, (0, 0, 255), -1)

                        x, y = kps_right[0, j, 0] * dim, kps_right[0, j, 1] * dim
                        x_fixed, y_fixed = kps_right_fixed[0, j, 0] * dim, kps_right_fixed[0, j, 1] * dim
                        img_kf[1] = cv2.circle(img_kf[1], (int(x), int(y)), 3, (0, 255, 0), -1)
                        img_kf[1] = cv2.circle(img_kf[1], (int(x_fixed), int(y_fixed)), 3, (0, 0, 255), -1)
                
                    kps_left, kps_right = kps_left_fixed, kps_right_fixed

                sizes.append(detections['left'][box_id_left].bb_size) 

                transformerInput[sample_idx, :, 0:2] = kps_left
                transformerInput[sample_idx, :, 2:4] = kps_right

            if torch.sum(transformerInput[:, -num_kps:, :4]) == 0:
                valid = False

            if not valid: 
                if found:
                    pred_pose, pred_kps = prev_pred_pose, prev_pred_kps
                else:
                    continue
            else:
                with torch.no_grad():
                    pred_pose, pred_kps = pose_estimator(transformerInput)

                yolo_pose_fps = 1/(time.time()-t2)
                yolo_pose_fps_list.append(yolo_pose_fps)

        else: # optimization case
            total_kps = {'left': [], 'right': []}
            m_labels = []
            mergable_params = []
            for sample_idx, (box_id_left, box_id_right, m_label) in enumerate(matches):
                
                kps_left = detections['left'][box_id_left].get_kps(detections['right'][box_id_right].kps, detections['right'][box_id_right].conf)
                kps_right = detections['right'][box_id_right].get_kps(kps_left, detections['left'][box_id_left].conf)
                sizes.append(detections['left'][box_id_left].bb_size) 

                m_labels.append(m_label)
                total_kps['left'].append(kps_left)
                total_kps['right'].append(kps_right)
                camR, camT = preprocessor.camR, preprocessor.camT
                mergable_params.append(detections['left'][box_id_left].objParams) # the assumption is that the Trackable will take the pose from left and right

            total_kps['left'] = torch.cat(total_kps['left'], dim=0)
            total_kps['right'] = torch.cat(total_kps['right'], dim=0)
            prevObjParams = ParamGroup(device=device, mergable_params=mergable_params)
            # meshes_to_optimize = [mesh for i, mesh in enumerate(all_meshes) if i+1 in m_labels]
            meshes_to_optimize = [all_meshes[m_label-1] for m_label in m_labels]
            optimizer = PoseOptimizer(device, image_pair, camR, camT, fx * dim, cx * dim, kps=total_kps, objects=meshes_to_optimize, numKps=12, frame_num=frame_num, prevObjParams=prevObjParams, earlyStop=True)
            t3 = time.time()
            poseParams, _, _, _ = optimizer.optimize()
            
            pose_fps = 1/(time.time()-t3)
            pose_fps_list.append(pose_fps)

            yolo_pose_fps = 1/(time.time()-t2)
            yolo_pose_fps_list.append(yolo_pose_fps)

            # split the params again and update the trackables
            for  sample_idx, (box_id_left, box_id_right, m_label) in enumerate(matches):
                # make dict of the params
                singleParams = {
                    'R': poseParams.R[sample_idx].unsqueeze(0),
                    't': poseParams.t[sample_idx].unsqueeze(0),
                    'a': [poseParams.a[sample_idx]],
                    'axis': [poseParams.axis[sample_idx]],
                    'isXArticulated': [poseParams.isXArticulated[sample_idx]],
                    'isYArticulated': [poseParams.isYArticulated[sample_idx]]
                }
                singleParamGroup = ParamGroup(device=device, single_params=singleParams, first=False)      
                detections['left'][box_id_left].objParams = singleParamGroup
                detections['right'][box_id_right].objParams = singleParamGroup  

            pred_pose = poseParams.paramToPose()

        for i, camera in enumerate(['left', 'right']):
            if args.visualize:
                cv2.imshow(f'{camera} yolo', yolo_plots[i])
            if args.save:
                cv2.imwrite(f'output/{experiment_name}/{sequence_name}/yolo_{camera}/{frame_num}.jpg', yolo_plots[i])
        cv2.waitKey(1)


        out2d, out3d = get_mesh(m_labels, pred_pose, all_meshes, cam_ext, cam_int, device)
        sampling_indices = [all_meshes[m_label-1].sampledPoints for m_label in m_labels]
        errors = []
        if use_transformer:
            for sample_idx in range(len(out2d[0])):
                error = 0
                reprojection_error = 0
                for c in range(2):
                    pred_yolo_kps = transformerInput[sample_idx, :, 2*c:2*(c+1)]
                    err = mpjpe(out2d[c][sample_idx][0, sampling_indices[sample_idx]], pred_yolo_kps * dim).item() 
                    error += (err / (sizes[sample_idx] * dim)) * 100 # percentage
                    reprojection_error += err
                
                reprojection_error /= 2
                error /= 2

                errors.append(error)
                reprojection_errors.append(reprojection_error)
            
            if  error > args.error_threshold:
                pred_pose, pred_kps = prev_pred_pose, prev_pred_kps
            else:
                found = True
                prev_pred_pose, prev_pred_kps = pred_pose, pred_kps
        else:
            found=True
            for sample_idx in range(len(out2d[0])):
                error = 0
                reprojection_error = 0
                for c in range(2):
                    pred_yolo_kps = total_kps['left'][sample_idx] if c == 0 else total_kps['right'][sample_idx]
                    err = mpjpe(out2d[c][sample_idx][0, sampling_indices[sample_idx]], pred_yolo_kps * dim).item() 
                    error += (err / (sizes[sample_idx] * dim)) * 100 
                    reprojection_error += err
                
                error /= 2
                reprojection_error /= 2

                errors.append(error)
                reprojection_errors.append(reprojection_error)

        if not found:
            if args.save:
                for i, camera in enumerate(['left', 'right']):
                    cv2.imwrite(f'output/{experiment_name}/{sequence_name}/yolo_{camera}/{frame_num}.jpg', yolo_plots[i])
            continue
        
        if error < args.error_threshold and args.save:
            visualize_meshes(out3d, out2d, all_meshes, m_labels, sequence_name, frame_num, 
                            imagePair=[np.copy(image_pair['left']), np.copy(image_pair['right'])], 
                            experiment_name=experiment_name, save_img=args.save, save_obj=args.save_objs)

        cv2.waitKey(1)  

    print('Average reprojection error (in pixels): ', sum(reprojection_errors) / len(reprojection_errors))
    print('Average YOLO + Pose FPS: ', sum(yolo_pose_fps_list) / len(yolo_pose_fps_list))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/varjo/')
    parser.add_argument('--sequence_name', type=str, default='seq1')
    parser.add_argument('--model_extension', type=str, default='pt', help='pt or engine (engine is much faster)')
    parser.add_argument('--model_size', type=str, default='m')
    parser.add_argument('--resolution', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fix_kps', action='store_true', help='Uses previous points and points from other frame to fix keypoints')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer for pose estimation')
    parser.add_argument('--save', action='store_true', help='Saves outputs to output folder')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_objs', action='store_true', help='Saves meshes to output folder')
    parser.add_argument('--error_threshold', type=int, default=10, help='Reprojection error threshold in %')

    args = parser.parse_args()

    predict_sequence(args)
