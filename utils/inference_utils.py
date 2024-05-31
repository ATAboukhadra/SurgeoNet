import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
from utils.one_euro_filter import OneEuroFilterKeypoints
from utils.mesh_utils import write_obj
from utils.params import ParamGroup


colors = [	(150, 52, 234), (84,197,224), (163,235,231), (240,246,246), (255,175,215),  \
            (67,42,18), (46,84,108), (77,166,124), (119,228,76), (202,255,0), \
            (255, 193, 77), (190,166,148), (122,30,71) 
            ]
def visualize_meshes(out3d, out2d, all_meshes, m_labels, sequenceName, frame_num, imagePair, experiment_name, save_obj=False, save_img=True):

    # disable numbers on axis
    sides = ['left', 'right']
    for c, frame in enumerate(imagePair):
        plt.figure(figsize=(11.52, 11.52), dpi=100)
        plt.axis('off')
        plt.imshow(frame)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for i in range(len(out3d)):

            mesh = all_meshes[m_labels[i] - 1]
            faces = mesh.faces[0].cpu().detach().numpy()
            verts = out3d[i][0].cpu().detach().numpy()
            verts[:, 0] *= -1
            verts[:, 1] *= -1

            verts2d = out2d[c][i][0].cpu().detach().numpy()
            x2d, y2d = verts2d[:, 0], verts2d[:, 1]

            r, g, b = colors[m_labels[i] - 1]

            plt.triplot(x2d, y2d, triangles=faces, color=(r/255, g/255, b/255), alpha=0.7)

            if save_obj:
                write_obj(verts, faces, f'output/{experiment_name}/{sequenceName}/objs/{frame_num}_{m_labels[i]}', pred=False)

        plt.tight_layout()
        plt.savefig(f'output/{experiment_name}/{sequenceName}/pose_{sides[c]}/{frame_num}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

        
def detect_flipping(kps, conf, label, kps_other_view=None, other_view_conf=None):
    flip_idx = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    found_flip = False
    if label in [2, 3, 4, 7, 13]:
        kp_left = kps[:, 10, 0] # bottom left
        kp_right = kps[:, 11, 0] # bottom right
    elif label in [5, 6]:
        kp_left = kps[:, 0, 0] # top left
        kp_right = kps[:, 1, 0] # top right
    elif label in [8, 11]:
        kp_left = kps[:, 0, 0] # top left
        kp_right = kps[:, 10, 0] # bottom left
    elif kps_other_view is not None: # 1 9 10 12
        # all corresponding keypoints from left and right views should have the same y value
        kps_y = kps[:, :, 1]
        kps_other_view_y = kps_other_view[:, :, 1]
        
        diff = abs(kps_y - kps_other_view_y)
        diff_flip = abs(kps_y - kps_other_view_y[:, flip_idx])

        diff = diff.mean()
        diff_flip = diff_flip.mean()
        if diff_flip < diff and conf < other_view_conf:
            found_flip = True
    else:
        return kps
        
    if found_flip or (label in [2, 3, 4, 5, 6, 7, 8, 11, 13] and kp_left > kp_right):
        kps = kps[:, flip_idx]

    return kps


class Trackable:
    def __init__(self, bbox, kps, label, bb_size, conf, num_kps, device):
        self.labels = [label]
        self.bbox = bbox
        self.prev_kps = None
        self.kps = kps
        self.bb_size = bb_size
        self.timestamp = 0
        self.conf = conf
        self.filter = OneEuroFilterKeypoints(num_values=2 * num_kps, freq=30, mincutoff=0.0001, beta=100.0, dcutoff=1.0)
        self.isFiltered = False
        yArticulatedLabels = [1, 2, 3, 4, 10, 11, 13]

        self.objParams = ParamGroup(device, isXArticulated = [label in [5, 6]], isYArticulated = [label in yArticulatedLabels], numObjects=1, isObject=True, prevParams=None)

    def update(self, bbox, kps, label, bb_size, conf):
        self.labels.append(label)
        self.bbox = bbox
        # copy the previous keypoint
        self.prev_kps = self.kps.detach().clone()
        self.kps = kps
        self.bb_size = bb_size
        self.conf = conf
        self.timestamp += 1 / 30.
        self.isFiltered = False

    def get_label(self):
        return max(set(self.labels), key=self.labels.count)


    def get_kps(self, other_view_kps, other_view_conf):
        m_label = self.get_label()
        self.kps = detect_flipping(self.kps, self.conf, m_label, other_view_kps, other_view_conf)
        if self.isFiltered:
            return self.filtered_kps
        else:
            self.filtered_kps = self.filter.filter(self.kps, self.timestamp)
            self.isFiltered = True

        return self.filtered_kps

def find_matches(detections_left, detections_right, new_ids):
    matches = []
    for box_id_left in new_ids['left']:
        left_top_left = detections_left[box_id_left].bbox[0, 1]
        left_bottom_left = detections_left[box_id_left].bbox[1, 1]
        dist = 0.01
        for box_id_right in new_ids['right']:
            right_top_left = detections_right[box_id_right].bbox[0, 1]
            right_bottom_left = detections_right[box_id_right].bbox[1, 1]

            pair_dist = abs(left_top_left - right_top_left) + abs(left_bottom_left - right_bottom_left)
            if pair_dist < dist:
                history = detections_left[box_id_left].labels + detections_right[box_id_right].labels
                m_label = max(set(history), key=history.count)
                matches.append((box_id_left, box_id_right, m_label))

    return matches


def check_validity(left_kps, right_kps):
    range1 = list(range(0, 12, 2))
    range2 = list(range(1, 12, 2))
    dist_left = (left_kps[0, range1] - left_kps[0, range2]).pow(2).sum(1).sqrt().max() * 1152
    dist_right = (right_kps[0, range1] - right_kps[0, range2]).pow(2).sum(1).sqrt().max() * 1152

    if dist_left / dist_right < 0.5:
        return False, True
    elif dist_right / dist_left < 0.5:
        return True, False
    
    return True, True

def fixer(left_detection, right_detection):
    kps_left = left_detection.kps.detach().clone()
    kps_right = right_detection.kps.detach().clone()
    is_fixed = False
    valid_left, valid_right = check_validity(kps_left, kps_right)
    if not valid_left and valid_right and left_detection.prev_kps is not None:
        prev_kps = left_detection.prev_kps.detach().clone()
        y_diff = kps_right[0, :, 1] - prev_kps[0, :, 1]
        prev_kps[0, :, 1] += y_diff
        kps_left = prev_kps
        is_fixed = True

    elif not valid_right and valid_left and right_detection.prev_kps is not None:
        prev_kps = right_detection.prev_kps.detach().clone()
        y_diff = kps_left[0, :, 1] - prev_kps[0, :, 1]
        prev_kps[0, :, 1] += y_diff
        kps_right = prev_kps
        is_fixed = True
    
    return kps_left, kps_right, is_fixed

def init_outputs(image_pair, experiment_name, sequence_name, frame_num):

    os.makedirs(f'output/{experiment_name}/{sequence_name}/frames_left', exist_ok=True)
    os.makedirs(f'output/{experiment_name}/{sequence_name}/frames_right', exist_ok=True)
    os.makedirs(f'output/{experiment_name}/{sequence_name}/pose_left', exist_ok=True)
    os.makedirs(f'output/{experiment_name}/{sequence_name}/pose_right', exist_ok=True)
    os.makedirs(f'output/{experiment_name}/{sequence_name}/yolo_left', exist_ok=True)
    os.makedirs(f'output/{experiment_name}/{sequence_name}/yolo_right', exist_ok=True)
    os.makedirs(f'output/{experiment_name}/{sequence_name}/objs', exist_ok=True)

    cv2.imwrite(f'output/{experiment_name}/{sequence_name}/frames_left/{frame_num}.png', image_pair['left'][:, :, ::-1])
    cv2.imwrite(f'output/{experiment_name}/{sequence_name}/frames_right/{frame_num}.png', image_pair['right'][:, :, ::-1])
    #copy the image to all the directories
    cv2.imwrite(f'output/{experiment_name}/{sequence_name}/pose_left/{frame_num}.png', cv2.resize(image_pair['left'][:, :, ::-1], (1122, 1122)))
    cv2.imwrite(f'output/{experiment_name}/{sequence_name}/pose_right/{frame_num}.png', cv2.resize(image_pair['right'][:, :, ::-1], (1122, 1122)))
    cv2.imwrite(f'output/{experiment_name}/{sequence_name}/yolo_left/{frame_num}.jpg', image_pair['left'][:, :, ::-1])
    cv2.imwrite(f'output/{experiment_name}/{sequence_name}/yolo_right/{frame_num}.jpg', image_pair['right'][:, :, ::-1])
