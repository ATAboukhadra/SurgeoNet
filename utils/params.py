import torch
import torch.nn as nn

class ParamGroup():
    def __init__(self, device, isObject=True, numObjects=1, isXArticulated=[], isYArticulated=[], nComps=6, prevParams=None, mergable_params=None, single_params=None, first=True) -> None:
        self.device = device
        self.nComps = nComps
        self.isObject = isObject
        self.axis = []
        self.first = first
        if mergable_params is not None:
            self.initMergableParams(mergable_params)
        elif single_params is not None:
            self.initSingleParams(single_params)
        else:
            self.numObjects = numObjects
            self.isXArticulated = isXArticulated
            self.isYArticulated = isYArticulated
            self.prevParams = prevParams
            if isObject:
                self.initObjectParams(prevParams)
            else:
                self.initHandParams(prevParams)
    
    def initSingleParams(self, params):
        self.numObjects = 1
        self.R = params['R']
        self.t = params['t']
        self.a = params['a']
        self.axis = params['axis']
        self.isXArticulated = params['isXArticulated']
        self.isYArticulated = params['isYArticulated']


    def initMergableParams(self, params):
        # this assumes a list of ParamGroup objects each with 1 object
        self.numObjects = len(params)
        self.R = torch.stack([param.R[0] for param in params], dim=0).to(self.device)
        self.t = torch.stack([param.t[0] for param in params], dim=0).to(self.device)
        self.a = [param.a[0] for param in params]
        self.axis = [param.axis[0] for param in params]
        self.isXArticulated = [param.isXArticulated[0] for param in params]
        self.isYArticulated = [param.isYArticulated[0] for param in params]
        # and all previous 
        self.first = any([param.first for param in params])

    def initObjectParams(self, prevParams):
        if self.prevParams is None:
            self.R = torch.tensor([[0.75, 0., 0]] * self.numObjects, requires_grad=True, device=self.device)
            self.t = torch.tensor([[[0.2], [0.1], [0.9]]] * self.numObjects, requires_grad=True, device=self.device)

            self.a = []
            for i in range(self.numObjects):
                if self.isXArticulated[i]:
                    self.axis.append(torch.tensor([-1.0, 0.0, 0.0], requires_grad=False, device=self.device).view(1, 3))
                elif self.isYArticulated[i]:
                    self.axis.append(torch.tensor([0.0, -1.0, 0.0], requires_grad=False, device=self.device).view(1, 3))
                else:
                    self.axis.append(torch.tensor([0.0, 0.0, 0.0], requires_grad=False, device=self.device).view(1, 3))
                
                if self.isXArticulated[i] or self.isYArticulated[i]:
                    self.a.append(torch.tensor(0.0, requires_grad=True, device=self.device))
                else:
                    self.a.append(torch.tensor(0.0, requires_grad=False, device=self.device))

        else:
            self.R = torch.empty_like(prevParams.R, requires_grad=True, device=self.device)
            self.t = torch.empty_like(prevParams.t, requires_grad=True, device=self.device)
            self.R.data.copy_(prevParams.R)
            self.t.data.copy_(prevParams.t)
            self.axis = prevParams.axis

            self.a = []
            for i in range(self.numObjects):
                isArticulated = self.isXArticulated[i] or self.isYArticulated[i]
                alpha = torch.empty_like(prevParams.a[i], requires_grad=isArticulated, device=self.device)
                alpha.data.copy_(prevParams.a[i])
                self.a.append(alpha)

    def initHandParams(self, prevParams):
        if self.prevParams is None:
            self.global_orient = torch.rand(self.numObjects, 3, requires_grad=True, device=self.device)
            self.pose = torch.rand(self.numObjects, self.nComps, requires_grad=True, device=self.device)
            self.betas = torch.rand(self.numObjects, 10, requires_grad=True, device=self.device)
            self.transl = torch.rand(self.numObjects, 3, requires_grad=True, device=self.device)
        else:
            self.global_orient = torch.empty_like(prevParams.global_orient, requires_grad=True, device=self.device)
            self.pose = torch.empty_like(prevParams.pose, requires_grad=True, device=self.device)
            self.betas = torch.empty_like(prevParams.betas, requires_grad=True, device=self.device)
            self.transl = torch.empty_like(prevParams.transl, requires_grad=True, device=self.device)

            self.global_orient.data.copy_(prevParams.global_orient)
            self.pose.data.copy_(prevParams.pose)
            self.betas.data.copy_(prevParams.betas)
            self.transl.data.copy_(prevParams.transl)

    def toList(self):
        if self.isObject:
            params = [self.R, self.t]
            params.extend(self.a)
            return params
        else:
            return [self.global_orient, self.pose, self.betas, self.transl]
    
    def temporalConsistencyLoss(self):
        t_loss = torch.tensor(0.0, device=self.device)
        w = 0.1
        if self.isObject and self.prevParams is not None:
            L1 = nn.L1Loss()
            t_loss += L1(self.R, self.prevParams.R) * w
            t_loss += L1(self.t, self.prevParams.t) * w
        
        return t_loss
    
    def alphaLoss(self):
        loss = torch.tensor(0.0, device=self.device)
        if self.isObject:
            w = 0.1
            for i in range(self.numObjects):
                if self.isXArticulated[i] or self.isYArticulated[i]:
                    loss += torch.max(torch.tensor(0, dtype=self.a[i].dtype), -self.a[i]) * w
        return loss

    def paramToPose(self):
        artiAxisBatch = torch.stack([self.axis[i] * self.a[i] for i in range(self.numObjects)], dim=0).squeeze(1)
        output = torch.cat([self.R, self.t.squeeze(-1), artiAxisBatch], dim=1)
        return output
