import torch
import cv2
from tqdm import tqdm
from utils.transform_utils import transform_points_batch, project_3D_points
from utils.pytorch3d_transforms import axis_angle_to_quaternion, quaternion_apply, axis_angle_to_matrix
from utils.params import ParamGroup

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def plotPoints(imagePair, leftPoints, rightPoints, color=(0, 0, 255), s=2):
    outLeft = imagePair['left'].copy()
    outRight = imagePair['right'].copy()
    for i in range(leftPoints.shape[0]):
        cv2.circle(outLeft, (int(leftPoints[i, 0]), int(leftPoints[i, 1])), s, color, -1)  # -1 for filled circle
        cv2.circle(outRight, (int(rightPoints[i, 0]), int(rightPoints[i, 1])), s, color, -1)  # -1 for filled circle

    return {'left': outLeft, 'right':outRight}

class PoseOptimizer(object):
    def __init__(self, device, imagePair, camR, camT, fVal, cVal, objects=[], numKps=50, objClasses=[], hands=[], nComps=6, frame_num=0,
                 visibility=None, masks=None, kps=None, silhouetteLoss=False, prevObjParams=None, prevHandParams=None, numIters=0, earlyStop=True) -> None:
        self.masks = masks
        self.kps = kps
        self.device = device
        self.objects = objects
        self.hands = hands
        self.numObjects = len(objects)
        self.numKps = numKps
        self.visibility = visibility
        if numIters == 0:
            self.numIters = 500 if (prevObjParams is None or prevObjParams.first) else 100
        else:
            self.numIters = numIters
        self.imagePair = imagePair
        self.frame_num = frame_num
        self.earlyStop = earlyStop
        self.initTransformVariables()
        self.initCamMtx(camR, camT, fVal, cVal)
        isXArticulated = [obj.isXArticulated for obj in objects]
        isYArticulated = [obj.isYArticulated for obj in objects]
        self.objParams = ParamGroup(device, isObject=True, isXArticulated=isXArticulated, isYArticulated=isYArticulated, numObjects=self.numObjects, nComps=nComps, prevParams=prevObjParams)
        self.handParams = ParamGroup(device, isObject=False, isXArticulated=[], isYArticulated=[], numObjects=len(hands), nComps=nComps, prevParams=prevHandParams)
        params = self.objParams.toList() + self.handParams.toList()
        self.optimizer = torch.optim.Adam(params, lr=0.01)

    def initTransformVariables(self):
        self.factor = torch.pi / 180.0
        self.yAxis = torch.tensor([0, -1, 0], device=self.device, dtype=torch.float32).view(1, 3)
        self.row = torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)
    
    def initCamMtx(self, camR, camT, fVal, cVal):
        self.camExt = {}
        for camera in ['left', 'right']:
            camExt = torch.cat((camR[camera], camT[camera].unsqueeze(-1)), dim=2).to(torch.float32)
            camExt = torch.cat((camExt, self.row.unsqueeze(0)), dim=1)
            self.camExt[camera] = camExt
            self.camInt = torch.tensor(
                [[fVal, 0., cVal],
                [0., fVal, cVal],
                [0., 0., 1.]], 
                dtype=torch.float32, device=self.device)
            
    def applyArticulation(self, j, verts, delim):
        articulation = torch.abs(self.objParams.a[j]) #* 30.0 * self.factor
        quatArti = axis_angle_to_quaternion(self.objParams.axis[j] * articulation).unsqueeze(1)
        numVerts = verts[:, delim:].shape[1]
        quatArtiMesh = quatArti.repeat(1, numVerts, 1)
        topPart = quaternion_apply(quatArtiMesh, verts[:, delim:])
        newVerts = torch.cat((verts[:, :delim], topPart), dim=1)
        return newVerts

    def createTransformationMatrix(self, j):
        R_mtx = axis_angle_to_matrix(self.objParams.R[j])# * 360.0 * self.factor)
        mtx = torch.cat((R_mtx, self.objParams.t[j]), dim=1)
        mtx = torch.cat((mtx, self.row), dim=0).unsqueeze(0)
        return mtx

    def project(self, side, vertsWorld):
        vertsCam = transform_points_batch(self.camExt[side], vertsWorld)
        mesh2d = project_3D_points(self.camInt, vertsCam, is_OpenGL_coords=True)
        return mesh2d
    
    def applyObjPose(self):
        objVerts2d = {
            'left': torch.zeros((self.numObjects, self.numKps, 2), device=self.device),
            'right': torch.zeros((self.numObjects, self.numKps, 2), device=self.device),
        }
        vertsWorldList = []
        for j, obj in enumerate(self.objects):
            verts = obj.verts
            delim = obj.delimiter
            sampledPoints = obj.sampledPoints
            newVerts = self.applyArticulation(j, verts, delim)
            mtx = self.createTransformationMatrix(j)
            vertsWorld = transform_points_batch(mtx, newVerts)
            vertsWorldList.append(vertsWorld)
            for camera in ['left', 'right']:
                verts2d = self.project(camera, vertsWorld)
                objVerts2d[camera][j] = verts2d[:, sampledPoints][0]
        
        objVerts2d = {camera: mesh.view(-1, 2) for camera, mesh in objVerts2d.items()}

        return objVerts2d
    
    def visualize(self, visiblePoints, hiddenPoints, i):

        outputPair = plotPoints(self.imagePair, visiblePoints['left'], visiblePoints['right'])
        if hiddenPoints is not None:
            outputPair = plotPoints(outputPair, hiddenPoints['left'], hiddenPoints['right'], color=(255, 0, 0), s=2)

        cv2.imshow('left optim', outputPair['left'][:, :, ::-1])
        if i+1 == self.numIters:
            cv2.imwrite(f'/home/aboukhadra/data/optim_full/left_{self.frame_num}.png', outputPair['left'][:, :, ::-1])
        cv2.imshow('right optim', outputPair['right'][:, :, ::-1])
        cv2.waitKey(1)

            
    def getObjMtx(self):
        objMtx = []
        for i in range(self.numObjects):
            mtx = self.createTransformationMatrix(i)
            objMtx.append(mtx)
        return objMtx
    
    def getQuatArti(self):
        quatArti = []
        for i in range(self.numObjects):
            articulation = torch.abs(self.objParams.a[i]) * 30.0 * self.factor
            quatArti.append(axis_angle_to_quaternion(self.objParams.axis[i] * articulation).unsqueeze(1))
        return quatArti

    def kpLoss(self, projKps):
        loss = torch.tensor(0.0, device=self.device)
        w = 1
        for camera in ['left', 'right']:            
            loss += mpjpe(projKps[camera].view(-1, self.numKps, 2), self.kps[camera] * 1152) * w
        loss /= 2
        return loss


    def optimize(self):

        for i in tqdm(range(self.numIters)):
            self.optimizer.zero_grad()
            objVerts2d = self.applyObjPose()

            loss = torch.tensor(0.0, device=self.device)
            if self.kps is not None:
                kpLoss = self.kpLoss(objVerts2d)
                loss += kpLoss

            if self.kps is not None and loss < 4 * self.kps['left'].shape[0] and self.earlyStop:
                break
            loss += self.objParams.temporalConsistencyLoss()
            loss += self.objParams.alphaLoss()
            loss.backward()
            self.optimizer.step()

        return self.objParams, self.handParams, i, objVerts2d
    