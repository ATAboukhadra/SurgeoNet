import numpy as np
import os
import torch
from PIL import Image


def readBinary(path, image, channels=4):
    """
    args:
        path (str): path of the image
        image (PIL.Image): empty PIL image
        channels (int): number of channels in the image (4:RGBA, 3:RGB)

    return:
        image (PIL.Image): RGB image read from binary file
    """

    # write a more efficient version of the above code
    with open(path, 'rb') as file:
        arr = bytearray(file.read())
    
    if image.width * image.height * channels != len(arr):
        raise ValueError("The image provided has the wrong size!")
    
    arr = np.array(arr).reshape(image.height, image.width, channels)
    arr = arr[:, :, :3]
    image = Image.fromarray(arr, 'RGB')
    return image

def read_data(filename):
    data = []

    with open(filename) as f:
        next(f) # skip header
        for line in f:
            data.append(float(line))
    return np.asarray(data)

def loadCalibrationVarjo(root):
    extrs = {}
    intrs = {}

    for view in ['Left', 'Right']:
        extr_path = os.path.join(root, f'{view}CameraExtrinsics.txt')
        intr_path = os.path.join(root, f'{view}CameraIntrinsics.txt')
        extr = read_data(extr_path).reshape(4,4).T
        R, t = extr[:3, :3], extr[:3, 3]
        extrs[view] = (R, t)

        intr = read_data(intr_path)
        intrs[view] = intr

    return extrs, intrs

def read_ext(filename, device):
    # Read the file that contains 17 lines, skip the first one and then create a 4x4 matrix out of the 16 values in the file

    with open(filename) as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [float(x) for x in lines]
        lines = np.array(lines)
        lines = lines.reshape((1, 4, 4))
        lines = torch.from_numpy(lines).transpose(2, 1).to(device).to(torch.float32)
        lines[:, :3, 3] *= (-1 if 'Right' in filename else 1)

        return lines[:, :3, :3], lines[:, :3, 3], lines