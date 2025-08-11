import torch
from torch import nn
from torchvision.ops import RoIAlign


class ROIAlign3d(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign3d, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.roi_align = RoIAlign(output_size, spatial_scale, sampling_ratio)

    def forward(self, input, rois):
        # RoIs should be in the format (batch_index, x1, y1, x2, y2)
        # Convert 3D RoIs to 2D by flattening the depth dimension
        bs, ch, d, h, w = input.shape
        input_2d = input.view(bs, ch * d, h, w)
        output = self.roi_align(input_2d, rois)
        # Reshape the output back to 3D
        output = output.view(output.size(0), ch, d, self.output_size[0], self.output_size[1])
        return output


    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
