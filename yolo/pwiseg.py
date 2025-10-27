import json
import numpy as np
import os
from tqdm import tqdm

root = '../data/pwiseg/'
categories = None
splits = ['train', 'val', 'test']

for split in splits:
    directory = root + split + '_dataset/'
    with open(directory + f'{split}.json') as f:
        data = json.load(f)
        if categories is None:
            categories = {}
            for i, category in enumerate(data['categories']):
                id = category['id']
                name = category['name']
                # keypoints = category['keypoints']
                skeleton = category['skeleton']
                # categories[id] = {'name': name, 'keypoints': keypoints, 'skeleton': skeleton}
                categories[id] = {'name': name, 'skeleton': skeleton}

        annotations = data['annotations']
        image_annotations = {}
        for sample in annotations:
            image_id = sample['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            
            category_id = sample['category_id']
            bbox = sample['bbox']        
            # keypoints = sample['keypoints']
            # print(bbox, keypoints)
            
            category = categories[category_id]
            # category_keypoints = category['keypoints']

            # full_keypoints = np.zeros((20, 3))
            idx = 0

            # for keypoint_idx in category_keypoints:
            #     x = keypoints[idx]
            #     y = keypoints[idx + 1]
            #     conf = keypoints[idx + 2]
            #     full_keypoints[int(keypoint_idx) - 1, 0] = x
            #     full_keypoints[int(keypoint_idx) - 1, 1] = y
            #     full_keypoints[int(keypoint_idx) - 1, 2] = conf
            #     idx += 3
            
            image_annotations[image_id].append([category_id, bbox])
            # image_annotations[image_id].append([category_id, bbox, full_keypoints])

        images = data['images']
        for image in images:
            image_id = image['id']
            image_annotation = image_annotations[image_id]
            W, H = image['width'], image['height']
            file_name = image['file_name'][:-4]
            
            # normalize bbox and keypoints
            for i in range(len(image_annotation)):
                # category_id, bbox, keypoints = image_annotation[i]
                category_id, bbox = image_annotation[i]
                
                x, y, w, h = bbox
                x = x + w / 2
                y = y + h / 2
                bbox = [x / W, y / H, w / W, h / H]
                # keypoints[:, 0] /= W
                # keypoints[:, 1] /= H
                
                image_annotation[i] = [category_id, bbox]

            os.makedirs(directory + f'labels/', exist_ok=True)
            with open(directory + f'labels/{file_name}.txt', 'w') as f:
                for i in range(len(image_annotation)):
                    # category_id, bbox, keypoints = image_annotation[i]
                    # f.write(f'{category_id} {" ".join(map(str, bbox))} {" ".join(map(str, keypoints.flatten()))}\n')
                    category_id, bbox = image_annotation[i]
                    f.write(f'{category_id} {" ".join(map(str, bbox))}\n')


        