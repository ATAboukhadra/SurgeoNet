from OneEuroFilter import OneEuroFilter
import torch

class OneEuroFilterKeypoints:
    def __init__(self, num_values=12, freq=30, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self._filters = []
        for _ in range(num_values):
            self._filters.append(OneEuroFilter(freq, mincutoff, beta, dcutoff))

    def filter(self, keypoints, timestamp):
        keypoints = keypoints.view(1, -1)
        out_keypoints = torch.zeros_like(keypoints).to(keypoints.device)

        for i in range(keypoints.shape[1]):
            out_keypoints[0, i] = self._filters[i](keypoints[0, i], timestamp)
        
        out_keypoints = out_keypoints.view(1, -1, 2)
        return out_keypoints
