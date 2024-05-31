import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os

from utils.pytorch3d_transforms import quaternion_apply, axis_angle_to_matrix, axis_angle_to_quaternion, rotation_6d_to_matrix
from utils.transform_utils import transform_points_batch, project_3D_points
from utils.mesh_utils import load_objects
from utils.varjo_utils import read_ext
from utils.optimizer import mpjpe

num_kps = 12

def iou(verts2d_gt, verts2d_pred):
    # verts2d_gt: N x 2
    # bounding box
    # downsample point clouds to 1000 points
    verts2d_gt = verts2d_gt[torch.randperm(verts2d_gt.shape[0])[:100]]
    verts2d_pred = verts2d_pred[torch.randperm(verts2d_pred.shape[0])[:100]]

    x1_gt, y1_gt = verts2d_gt.min(dim=0)[0]
    x2_gt, y2_gt = verts2d_gt.max(dim=0)[0]
    x1_pred, y1_pred = verts2d_pred.min(dim=0)[0]
    x2_pred, y2_pred = verts2d_pred.max(dim=0)[0]

    # iou loss 
    xA = torch.max(x1_gt, x1_pred)
    yA = torch.max(y1_gt, y1_pred)
    xB = torch.min(x2_gt, x2_pred)
    yB = torch.min(y2_gt, y2_pred)
    interArea = torch.max(torch.tensor(0.0, device=verts2d_gt.device), xB - xA + 1) * torch.max(torch.tensor(0.0, device=verts2d_gt.device), yB - yA + 1)
    boxAArea = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
    boxBArea = (x2_pred - x1_pred + 1) * (y2_pred - y1_pred + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)
    return 1 - iou.mean()

def train(model, train_loader, val_loader, start_epoch, num_epochs, lr, device, model_name):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    all_meshes, translation_dict = load_objects(device, range(1, 14), nKps=num_kps)
    cam_ext, cam_int = load_cam_mat(device)

    best_error = 1e5
    for epoch in range(start_epoch, num_epochs+1):
        running_pose_loss = 0.0
        running_mesh_loss = 0.0
        running_kps_loss = 0.0
        running_iou_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, pose, kps = data
            optimizer.zero_grad()

            pred_pose, pred_kps = model(inputs.to(device))
            if len(inputs.shape) == 3:
                object_labels = inputs[:, 0, 4:18].argmax(dim=1)
            else:
                object_labels = inputs[:, 0, 0, 4:18].argmax(dim=1)

            pose = pose.to(device)
            verts2d_gt, verts_gt = get_mesh(object_labels, pose, all_meshes, cam_ext, cam_int, device)
            verts2d_pred, verts_pred = get_mesh(object_labels, pred_pose, all_meshes, cam_ext, cam_int, device)

            # calculate loss
            max_value = torch.max(torch.abs(pose[:, -3:-1]), dim=1)[0]
            indices = max_value > 0.0001
            arti_loss = criterion(pred_pose[indices, -3:-1], pose[indices, -3:-1])

            pose_loss = criterion(pred_pose[:, :-3], pose[:, :-3])
            pose_loss += arti_loss
            
            mesh_loss = torch.tensor(0.0, device=device)
            iou_loss = torch.tensor(0.0, device=device)

            for j in range(len(verts_gt)):
                mesh_loss += mpjpe(verts_gt[j], verts_pred[j])
                if 'iou' in model_name:
                    iou_loss += (iou(verts2d_gt[0][j][0], verts2d_pred[0][j][0]) + iou(verts2d_gt[1][j][0], verts2d_pred[1][j][0])) / 2

            mesh_loss /= len(verts_gt)
            iou_loss /= len(verts_gt)
            kps_loss = mpjpe(pred_kps[:, -num_kps:], kps.to(device))

            if torch.isnan(pose_loss).any():
                loss = mesh_loss + kps_loss + iou_loss
            else:
                running_pose_loss += pose_loss.item()
                loss = mesh_loss + pose_loss + kps_loss + iou_loss

            loss.backward()
            optimizer.step()

            running_mesh_loss += mesh_loss.item()
            running_kps_loss += kps_loss.item()
            running_iou_loss += iou_loss.item()

            if (i+1) % 100 == 0:
                print('[%d, %5d] loss: 7D: %.4f, Mesh: %.4f, Kps: %.4f, IoU: %.4f' %
                      (epoch, i + 1, running_pose_loss / 100, running_mesh_loss / 100, running_kps_loss / 100, running_iou_loss / 100))
                running_pose_loss = 0.0
                running_mesh_loss = 0.0
                running_kps_loss = 0.0
                running_iou_loss = 0.0
        # Run validation
        error = evaluate(val_loader, model, device, model_name)

        # Save model
        if error < best_error:
            best_error = error
            os.makedirs(f'checkpoints/{model_name}', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/{model_name}/{epoch}.pth')
            print(f'Saved model to checkpoints/{model_name}/{epoch}.pth')

def evaluate(val_loader, model, device, model_name, half=False):
    criterion = nn.MSELoss()
    all_meshes, translation_dict = load_objects(device, range(1, 14), nKps=num_kps)
    cam_ext, cam_int = load_cam_mat(device)

    with torch.no_grad():
        running_pose_loss = 0.0
        running_kps_loss = 0.0
        running_mesh_loss = 0.0
        running_iou_loss = 0.0
        length = len(val_loader)
        for i, data in enumerate(val_loader, 0):
            if half and i == 100: break
            inputs, pose, kps = data
            pred_pose, pred_kps = model(inputs.to(device))

            if len(inputs.shape) == 3:
                object_labels = inputs[:, 0, 4:18].argmax(dim=1)
            else:
                object_labels = inputs[:, 0, 0, 4:18].argmax(dim=1)
            verts2d_gt, verts_gt = get_mesh(object_labels, pose.to(device), all_meshes, cam_ext, cam_int, device)
            verts2d_pred, verts_pred = get_mesh(object_labels, pred_pose, all_meshes, cam_ext, cam_int, device)
            # calculate loss
            pose_loss = criterion(pred_pose[:, :-4], pose.to(device)[:, :-4])

            # apply the articulation loss only on samples that have non zero articulation
            max_value = torch.max(torch.abs(pose[:, -3:-1]), dim=1)[0]
            indices = max_value > 0.0001
            arti_loss = criterion(pred_pose[indices, -3:-1], pose.to(device)[indices, -3:-1])
            
            pose_loss += arti_loss

            mesh_loss = torch.tensor(0.0, device=device)
            iou_loss = torch.tensor(0.0, device=device)
            for j in range(len(verts_gt)):
                mesh_loss += mpjpe(verts_gt[j], verts_pred[j])
                if 'iou' in model_name:
                    iou_loss += (iou(verts2d_gt[0][j][0], verts2d_pred[0][j][0]) + iou(verts2d_gt[1][j][0], verts2d_pred[1][j][0])) / 2

            iou_loss /= len(verts_gt)
            mesh_loss /= len(verts_gt)
            kps_loss = mpjpe(pred_kps[:, -num_kps:], kps.to(device))

            if not torch.isnan(pose_loss).any():
                running_pose_loss += pose_loss.item()

            running_kps_loss += kps_loss.item()
            running_mesh_loss += mesh_loss.item()
            running_iou_loss += iou_loss.item()

        print('Val loss: 7D: %.4f, Mesh: %.4f, Kps: %.4f, IoU: %.4f' %
                      (running_pose_loss / length, running_mesh_loss / length, running_kps_loss / length, running_iou_loss / length))
        return running_kps_loss / length

def collate_fn(batch):
    inputs = []
    poses = []
    kps = []
    for i, data in enumerate(batch):
        inputs.append(data[0])
        poses.append(data[1])
        kps.append(data[2])
    inputs = torch.cat(inputs, dim=0)
    poses = torch.cat(poses, dim=0)
    kps = torch.cat(kps, dim=0)
    return inputs, poses, kps

def get_mesh(object_labels, pose, all_meshes, cam_ext, cam_int, device):

    meshes = [all_meshes[o-1] for o in object_labels]
    if pose.shape[1] != 6 + 3 + 3:
        rot, transl, arti_axis = pose[:, :3], pose[:, 3:6].unsqueeze(-1), pose[:, 6:]
        quat_arti = axis_angle_to_quaternion(arti_axis)
        rotMat = axis_angle_to_matrix(rot)
    else:
        rot, transl, arti_axis = pose[:, :6], pose[:, 6:9].unsqueeze(-1), pose[:, 9:]
        quat_arti = axis_angle_to_quaternion(arti_axis)
        rotMat = rotation_6d_to_matrix(rot)

    transformation = torch.cat((rotMat, transl), dim=2)
    row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(transformation.shape[0], 1, 1)
    mat = torch.cat((transformation, row), dim=1)
    verts2d, verts3d = transform(meshes, quat_arti, mat, cam_ext, cam_int)

    return verts2d, verts3d

def transform(meshes, quat_arti, mat, cam_ext, cam_int):
    out2d = {0: [], 1: []}
    out3d = []
    for i, m in enumerate(meshes):
        if m.delimiter != m.verts.shape[1]:
            num_verts = m.verts[:, m.delimiter:].shape[1]
            quat_arti_mesh = quat_arti[i].repeat(1, num_verts, 1)
            top_part = quaternion_apply(quat_arti_mesh, m.verts[:, m.delimiter:])
            new_verts = torch.cat((m.verts[:, :m.delimiter], top_part), dim=1)
        else:
            new_verts = m.verts

        verts_world = transform_points_batch(mat[i].unsqueeze(0), new_verts)
        out3d.append(verts_world)

        for c in range(2):
            verts_cam = transform_points_batch(cam_ext[c], verts_world)
            # sample keypoints from verts2d unfirormly
            mesh2d = project_3D_points(cam_int, verts_cam, is_OpenGL_coords=True)
            out2d[c].append(mesh2d)

    return out2d, out3d

def getKps2d(verts3d, cam_ext, cam_int):
    out2d = []
    for c in range(2):
        verts_cam = transform_points_batch(cam_ext[c], verts3d[:1])
        # sample keypoints from verts2d unfirormly
        verts2d = project_3D_points(cam_int, verts_cam, is_OpenGL_coords=True)
        out2d.append(verts2d)
    
    return out2d

def load_cam_mat(device):
    _, _, cam_ext_left = read_ext('varjo_calibration/LeftCameraExtrinsics.txt', device)
    _, _, cam_ext_right = read_ext('varjo_calibration/RightCameraExtrinsics.txt', device)

    cam_ext = [cam_ext_left, cam_ext_right]

    size = 1152
    fx, fy, cx, cy = 0.65 * size, 0.65 * size, 0.5 * size, 0.5 * size
    cam_int = torch.tensor([[fx, 0, cx], 
                            [0, fy, cy], 
                            [0, 0, 1]], device=device)

    return cam_ext, cam_int

def visualize(model, loader, device):
    model.eval()
    all_meshes, translation_dict = load_objects(device, range(1, 14), nKps=num_kps)
    cam_ext, cam_int = load_cam_mat(device)

    # Get a single sample from the loader
    for i, data in enumerate(loader):
        inputs, pose, kps = data
        inputs = inputs.to(device)
        pred_pose, pred_kps = model(inputs)

        if len(inputs.shape) == 3:
            object_labels = inputs[:, 0, 4:18].argmax(dim=1)
            keypoints1 = inputs[0, :, :2].cpu().numpy()
            keypoints2 = inputs[0, :, 2:4].cpu().numpy()
        else: # temporal dimension
            object_labels = inputs[:, 0, 0, 4:18].argmax(dim=1)
            keypoints1 = inputs[0, -1, :, :2].cpu().numpy()
            keypoints2 = inputs[0, -1, :, 2:4].cpu().numpy()

        out, _ = get_mesh(object_labels, pred_pose, all_meshes, cam_ext, cam_int, device)

        # Create empty images to plot keypoints on
        img1 = np.zeros((1152, 1152, 3), dtype=np.uint8)
        img2 = np.zeros((1152, 1152, 3), dtype=np.uint8)

        # Plot keypoints on the images
        for i in range(keypoints1.shape[0]):
            x1, y1 = keypoints1[i, 0] * 1152, keypoints1[i, 1] * 1152
            x2, y2 = keypoints2[i, 0] * 1152, keypoints2[i, 1] * 1152
            img1 = cv2.circle(img1, (int(x1), int(y1)), 3, (163, 214, 245), -1)
            img2 = cv2.circle(img2, (int(x2), int(y2)), 3, (163, 214, 245), -1)

        # Plot the mesh on the images
        for i in range(out[0][0].shape[1]):
            x1, y1 = out[0][0][0, i, 0].item(), out[0][0][0, i, 1].item()
            x2, y2 = out[1][0][0, i, 0].item(), out[1][0][0, i, 1].item()
            img1 = cv2.circle(img1, (int(x1), int(y1)), 3, (255, 215, 0), -1)
            img2 = cv2.circle(img2, (int(x2), int(y2)), 3, (255, 215, 0), -1)

        if pred_kps is not None:
            kps2d = getKps2d(pred_kps, cam_ext, cam_int)
            for i in range(kps2d[0].shape[1]):
                x1, y1 = kps2d[0][0, i, 0], kps2d[0][0, i, 1]
                x2, y2 = kps2d[1][0, i, 0], kps2d[1][0, i, 1]
                img1 = cv2.circle(img1, (int(x1), int(y1)), 3, (205, 92, 92), -1)
                img2 = cv2.circle(img2, (int(x2), int(y2)), 3, (205, 92, 92), -1)

        # Display the images
        cv2.imshow('Image 1', img1[:, :, ::-1])
        cv2.imshow('Image 2', img2[:, :, ::-1])
        if cv2.waitKey(0) & 0xFF == ord('q'):
            continue
    cv2.destroyAllWindows()

