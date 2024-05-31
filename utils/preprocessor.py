import cv2
import torch
import numpy as np
import os
from PIL import Image
from utils.varjo_utils import readBinary, loadCalibrationVarjo

class Preprocessor():

    def __init__(self, device, root, dim):
        self.root = root
        self.device = device
        self.dim = dim
        self.num_channels = 3
        self.loadCamMtx(device)

    def loadCamMtx(self, device):
        extrs, intrs = loadCalibrationVarjo('varjo_calibration')
        self.extrs = extrs
        self.intrs = intrs
        fx, fy = 0.625, 0.625
        cx, cy = 0.5, 0.5
        self.Kn = np.array([[fx * self.dim, 0., cx * self.dim],
                [0., fy * self.dim, cy * self.dim],
                [0., 0., 1.]], np.float32)
        leftR, leftT = torch.tensor(extrs['Left'][0], device=device).unsqueeze(0), torch.tensor(extrs['Left'][1], device=device).unsqueeze(0)
        rightR, rightT = torch.tensor(extrs['Right'][0], device=device).unsqueeze(0), torch.tensor(extrs['Right'][1], device=device).unsqueeze(0)
        rightT *= -1
        self.camR = {'left': leftR, 'right': rightR}
        self.camT = {'left': leftT, 'right': rightT}

    
    def readPair(self, frameNum):
        imagePair = {}
        for camera in ['left', 'right']:
            filepath = os.path.join(self.root, f'varjo_{camera}_{frameNum}.bin')
            if not os.path.exists(filepath):
                return None
            image = Image.new('RGBA', (self.dim, self.dim))
            image = readBinary(filepath, image, self.num_channels)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
            imagePair[camera] = image

        return imagePair