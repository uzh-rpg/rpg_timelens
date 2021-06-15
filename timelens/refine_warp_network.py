import torch as th
from timelens.common import warp
from timelens import fusion_network, warp_network
from timelens.superslomo import unet

def _pack_for_residual_flow_computation(example):
    tensors = [
        example["middle"]["{}_warped".format(packet)] for packet in ["after", "before"]
    ]
    tensors.append(example["middle"]["fusion"])
    return th.cat(tensors, dim=1)


def _pack_images_for_second_warping(example):
    return th.cat(
        [example["middle"]["after_warped"], example["middle"]["before_warped"]],
    )


def _pack_output_to_example(example, output):
    (
        example["middle"]["before_refined_warped"],
        example["middle"]["after_refined_warped"],
        example["middle"]["before_refined_warped_invalid"],
        example["middle"]["after_refined_warped_invalid"],
        example["before"]["residual_flow"],
        example["after"]["residual_flow"],
    ) = output


class RefineWarp(warp_network.Warp, fusion_network.Fusion):
    def __init__(self):
        warp_network.Warp.__init__(self)
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        self.flow_refinement_network = unet.UNet(9, 4, False)

    def run_refine_warp(self, example):
        warp_network.Warp.run_and_pack_to_example(self, example)
        fusion_network.Fusion.run_and_pack_to_example(self, example)
        residual = self.flow_refinement_network(
            _pack_for_residual_flow_computation(example)
        )
        (after_residual, before_residual) = th.chunk(residual, 2, dim=1)
        residual = th.cat([after_residual, before_residual], dim=0)
        refined, refined_invalid = warp.backwarp_2d(
            source=_pack_images_for_second_warping(example),
            y_displacement=residual[:, 0, ...],
            x_displacement=residual[:, 1, ...],
        )
        
        (after_refined, before_refined) = th.chunk(refined, 2)
        (after_refined_invalid, before_refined_invalid) = th.chunk(
            refined_invalid.detach(), 2)
        return (
            before_refined,
            after_refined,
            before_refined_invalid,
            after_refined_invalid,
            before_residual,
            after_residual,
        )


    def run_fast(self, example):
        warp_network.Warp.run_and_pack_to_example(self, example)
        fusion_network.Fusion.run_and_pack_to_example(self, example)
        residual = self.flow_refinement_network(
            _pack_for_residual_flow_computation(example)
        )
        (after_residual, before_residual) = th.chunk(residual, 2, dim=1)
        residual = th.cat([after_residual, before_residual], dim=0)
        refined, _ = warp.backwarp_2d(
            source=_pack_images_for_second_warping(example),
            y_displacement=residual[:, 0, ...],
            x_displacement=residual[:, 1, ...],
        )

        return th.chunk(refined, 2)

    def run_and_pack_to_example(self, example):
        _pack_output_to_example(example, self.run_refine_warp(example))

    def forward(self, example):
        return self.run_refine_warp(example)
