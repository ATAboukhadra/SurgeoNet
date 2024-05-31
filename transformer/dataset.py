import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.mesh_utils import load_objects
from utils.pytorch3d_transforms import matrix_to_rotation_6d, quaternion_apply, matrix_to_axis_angle, quaternion_to_matrix
from utils.transform_utils import transform_points_batch, project_3D_points


def one_hot_encode(labels, num_classes=14):
    one_hot = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        one_hot[i][label] = 1
    return one_hot

class SyntheticStereoPoseDataset(Dataset):
    def __init__(self, split, mask=False, stereo=True, augment=False, 
                 dataset='', num_kps=12, rot6d=True, pos_embed=True, cls_embed=True):
        self.split = split
        self.labels_dir = f'data/{dataset}/{split}/labels'
        self.transformations_dir = f'data/{dataset}/{split}/transformations'
        self.samples = os.listdir(self.labels_dir)
        self.samples = self.unique_names()
        self.all_meshes, _ = load_objects('cpu', range(1, 14), nKps=num_kps)
        self.mask = mask
        self.stereo = stereo
        self.num_kps = num_kps
        self.rot6d = rot6d
        self.augment = augment
        self.pos_embed = pos_embed
        self.cls_embed = cls_embed

    def unique_names(self):
        unique = list(set([x.split('_')[0] for x in sorted(self.samples)]))
        invalid_samples = []
        # remove all samples if the there is a different number of lines in 0 and 1
        for sample in tqdm(unique):
            # if one of the files is missing, remove the sample
            if not os.path.exists(os.path.join(self.labels_dir, f'{sample}_0.txt')) or not os.path.exists(os.path.join(self.labels_dir, f'{sample}_1.txt')):
                invalid_samples.append(sample)
                continue
            kps_left = os.path.join(self.labels_dir, f'{sample}_0.txt')
            kps_right = os.path.join(self.labels_dir, f'{sample}_1.txt')
            with open(kps_left, 'r') as f:
                lines_left = f.readlines()
            with open(kps_right, 'r') as f:
                lines_right = f.readlines()
            if len(lines_left) != len(lines_right):
                invalid_samples.append(sample)
        
        unique = [x for x in unique if x not in invalid_samples]

        return unique
    
    def __len__(self):
        return len(self.samples)

    def get_kps(self, m, pose):
        # mesh is a Mesh object
        # pose is a 4x4 transformation matrix
        mat = pose[:16].view(1, 4, 4)
        quat_arti = pose[16:].unsqueeze(0)
        if m.delimiter != m.verts.shape[1]:
            num_verts = m.verts[:, m.delimiter:].shape[1]
            quat_arti_mesh = quat_arti.repeat(1, num_verts, 1)
            top_part = quaternion_apply(quat_arti_mesh, m.verts[:, m.delimiter:])
            new_verts = torch.cat((m.verts[:, :m.delimiter], top_part), dim=1)
        else:
            new_verts = m.verts

        verts_world = transform_points_batch(mat, new_verts)
        kps = verts_world[:, m.sampledPoints]

        return kps[0]
    
    def get_bbox(self, x_center, y_center, width, height):
        x_top_left, y_top_left = x_center - width/2, y_center - height/2
        x_bottom_right, y_bottom_right = x_center + width/2, y_center + height/2
        bbox = torch.tensor([[x_top_left, y_top_left], [x_bottom_right, y_bottom_right]])
        return bbox

    def augmentation(self, kps):
        pixel = 1 / 1152
        # add either 1 or 2 pixels to each keypoint in a random direction
        for i in range(kps.shape[0]):
            direction = torch.randint(0, 4, (1, ))[0]
            amount = torch.randint(0, 5, (1, ))[0]
            if direction == 0:
                kps[i, 0] += pixel * amount
            elif direction == 1:
                kps[i, 0] -= pixel * amount
            elif direction == 2:
                kps[i, 1] += pixel * amount
            elif direction == 3:
                kps[i, 1] -= pixel * amount

        return kps
    
    def augmentation_mid(self, kps):
        # this function should augment the keypoints by translating them by a percentage of the distance to the midpoint between the keypoint and its 
        # corresponding keypoint

        for i in range(kps.shape[0]):
            corr_i = (i+1) if i % 2 == 0 else (i-1)
            mid = (kps[i] + kps[corr_i]) / 2
            #sample percentage from distribution such that the bias is more towards the lower values and maximum value is 1
            amount = torch.pow(torch.rand(1), 2) 
            dir_vec = mid - kps[i]
            kps[i] += dir_vec * amount

        return kps
    
    
    def add_kp_class(self, input_batch):
        # this function is called in case of pos_embed or sym_embed
        num_classes = self.num_kps #if self.pos_embed else self.num_kps // 2 # in case of symmetric keypoints we only need half the number of classes
        # add classes [0, 0, 1, 1, ...] in case of symmetric keypoints
        classes = [i for i in range(num_classes)] #if self.pos_embed else [i // 2 for i in range(num_classes * 2)]
        one_hot = one_hot_encode(classes, num_classes=num_classes).unsqueeze(0).repeat(input_batch.shape[0], 1, 1)
        input_batch = torch.cat((input_batch, one_hot), dim=2)

        return input_batch

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        kps_left = os.path.join(self.labels_dir, f'{sample_name}_0.txt')
        kps_right = os.path.join(self.labels_dir, f'{sample_name}_1.txt')
        transformation_path = os.path.join(self.transformations_dir, f'{sample_name}_0.txt')
        samples = []
        labels = []
        transformations = []

        # binary flag for side augmentations
        side_aug = torch.rand(1) < 0.5
        # Load keypoints
        with open(kps_left, 'r') as f:
            for i, line in enumerate(f.readlines()):
                labels.append(int(line.split()[0]))
                kps = torch.tensor([float(x) for x in line.split()[5:]]).view(-1, 2)
                if self.augment:
                    kps = self.augmentation(kps)
                    if side_aug:
                        kps = self.augmentation_mid(kps)
                samples.append(kps)

        if self.stereo:
            with open(kps_right, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    kps = torch.tensor([float(x) for x in line.split()[5:]]).view(-1, 2)
                    if self.augment:
                        kps = self.augmentation(kps)
                        if not side_aug:
                            kps = self.augmentation_mid(kps)
                    samples[i] = torch.cat((samples[i], kps), dim=1)
        else:
            # append zeros to the right keypoints
            for i in range(len(samples)):
                samples[i] = torch.cat((samples[i], torch.zeros_like(samples[i])), dim=1)


        kps3d = []
        with open(transformation_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                pose = torch.tensor([float(x) for x in line.split()])
                mesh = self.all_meshes[labels[i]-1]
                kps3d.append(self.get_kps(mesh, pose))
                mat = pose[:16].view(1, 4, 4)
                if self.rot6d:
                    rot = matrix_to_rotation_6d(mat[:, :3, :3])[0]
                else:
                    rot = matrix_to_axis_angle(mat[:, :3, :3])[0]
                
                transl = mat[0, :3, 3]
                arti = pose[16:]
                arti = quaternion_to_matrix(arti.unsqueeze(0))
                arti = matrix_to_axis_angle(arti)[0]
                
                pose7d = torch.cat((rot, transl, arti))        
                transformations.append(pose7d)
        
        kps3d = torch.stack(kps3d)
        input_batch = torch.stack(samples)

        if self.mask:
            # disable all left or all right keypoints in 20% of the cases
            maskSample = torch.rand(1) < 0.2
            if maskSample:
                maskLeft = torch.rand(1) < 0.5
                if maskLeft:
                    input_batch[:, :, :2] = 0
                else:
                    input_batch[:, :, 2:] = 0

        # repeat the one-hot labels for each keypoint and concatenate
        if self.cls_embed:
            one_hot_labels = one_hot_encode(labels).unsqueeze(1).repeat(1, self.num_kps, 1)
            input_batch = torch.cat((input_batch, one_hot_labels), dim=2)
        else:
            # append zeros (handle the fact that you use this later to get the class)
            input_batch = torch.cat((input_batch, torch.zeros(input_batch.shape[0], self.num_kps, 14)), dim=2)

        if self.pos_embed:
            # add positional encoding to the input batch
            input_batch = self.add_kp_class(input_batch)
        else:
            # append zeros
            input_batch = torch.cat((input_batch, torch.zeros(input_batch.shape[0], self.num_kps, self.num_kps)), dim=2)
        output_pose = torch.stack(transformations)
        return input_batch, output_pose, kps3d