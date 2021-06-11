import torch as th
from timelens.superslomo import unet
from torch import nn
from timelens.common import warp


def _pack_voxel_grid_for_flow_estimation(example):
    return th.cat(
        [example["before"]["reversed_voxel_grid"], example["after"]["voxel_grid"]]
    )


def _pack_images_for_warping(example):
    return th.cat(
        [example["before"]["rgb_image_tensor"], example["after"]["rgb_image_tensor"]]
    )


def _pack_output_to_example(example, output):
    (
        example["middle"]["before_warped"],
        example["middle"]["after_warped"],
        example["before"]["flow"],
        example["after"]["flow"],
        example["middle"]["before_warped_invalid"],
        example["middle"]["after_warped_invalid"],
    ) = output


class Warp(nn.Module):
    def __init__(self):
        super(Warp, self).__init__()
        self.flow_network = unet.UNet(5, 2, False)

    def from_legacy_checkpoint(self, checkpoint_filename):
        checkpoint = th.load(checkpoint_filename)
        self.load_state_dict(checkpoint["networks"])

    def run_warp(self, example):
        flow = self.flow_network(_pack_voxel_grid_for_flow_estimation(example))
        warped, warped_invalid = warp.backwarp_2d(
            source=_pack_images_for_warping(example),
            y_displacement=flow[:, 0, ...],
            x_displacement=flow[:, 1, ...],
        )
        (before_flow, after_flow) = th.chunk(flow, chunks=2)
        (before_warped, after_warped) = th.chunk(warped, chunks=2)
        (before_warped_invalid, after_warped_invalid) = th.chunk(
            warped_invalid.detach(), chunks=2
        )
        return (
            before_warped,
            after_warped,
            before_flow,
            after_flow,
            before_warped_invalid,
            after_warped_invalid,
        )
    
    def run_and_pack_to_example(self, example):
        _pack_output_to_example(example, self.run_warp(example))

    def forward(self, example):
        return self.run_warp(example)
