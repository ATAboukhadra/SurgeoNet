import torch
from torch import nn
from ultralytics import YOLO

class StereoYOLOMultiInput(nn.Module):
    def __init__(self, model, max_preds=200, resolution=512):
        super().__init__()
        self.model = model
        self.max_preds = max_preds
        self.orig_shape = 1152
        self.resize_shape = resolution

    def scale_output(self, output):

        gain = self.resize_shape / self.orig_shape
        
        output[0:4] /= gain 
        output[18:18*2*12] /= gain

        return output

    def forward(self, x, y):
        # X is a tensor of size 1152x1152x3x2 flattened
        mid = x.shape[0] // 2
        image1 = x
        image2 = y
        
        # Reshape to 1x1152x1152x3
        image1 = image1.view(1, 1152, 1152, 3)
        image2 = image2.view(1, 1152, 1152, 3)

        # Flip the image on x axis
        image1 = torch.flip(image1, dims=[1])
        image2 = torch.flip(image2, dims=[1])
        
        # Channels first
        image1 = image1.permute(0, 3, 1, 2)
        image2 = image2.permute(0, 3, 1, 2)

        # Resize to 1x3x512x512
        image1 = nn.functional.interpolate(image1, size=(self.resize_shape, self.resize_shape))
        image2 = nn.functional.interpolate(image2, size=(self.resize_shape, self.resize_shape))

        x = torch.cat([image1, image2], dim=0)

        out = self.model(x)
        # out[0] is output and out[1] is features
        # out[1][0][i] is the multi-scale features for 3 levels # 78xNxN
        # out[1][1][i] is the still unknown for me but it has the size 24x27216 
        
        conf = out[0][:, 4:18, :]
        outputs_list = []

        # sort the predictions by max confidence without numpy
        for i in range(conf.shape[0]):
            max_conf, _ = torch.max(conf[i, :, :], dim=0)
            _, idx = torch.sort(max_conf, dim=0, descending=True)
            idx = idx[:self.max_preds]
            output = out[0][i, :, idx]
            output = self.scale_output(output)
            outputs_list.append(output)
        
        output = torch.stack(outputs_list, dim=0)
        return output


if __name__ == '__main__':
    # Load a model
    resolution = 640
    model = YOLO(f'runs/pose/s_{resolution}/weights/best.pt')

    model = StereoYOLOMultiInput(model.model, resolution=resolution).to('cuda')

    dummy_input1 = torch.randn(1152 * 1152 * 3).to('cuda')
    dummy_input2 = torch.randn(1152 * 1152 * 3).to('cuda')

    model.eval()

    torch.onnx.export(model, (dummy_input1, dummy_input2), "yolos_512.onnx", input_names=['input1', 'input2'])
