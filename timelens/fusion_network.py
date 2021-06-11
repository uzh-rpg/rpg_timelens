import torch as th
from timelens.superslomo import unet
from torch import nn

def _pack(example):
    return th.cat([example['before']['voxel_grid'],
                   example['before']['rgb_image_tensor'],
                   example['after']['voxel_grid'],
                   example['after']['rgb_image_tensor']], dim=1)


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        
    def run_fusion(self, example):
        return self.fusion_network(_pack(example))
        
    def from_legacy_checkpoint(self, checkpoint_filename):
        checkpoint = th.load(checkpoint_filename)
        self.load_state_dict(checkpoint["networks"])

    def run_and_pack_to_example(self, example):
        example['middle']['fusion'] = self.run_fusion(example)
        
    def forward(self, example):
        return self.run_fusion(example)
