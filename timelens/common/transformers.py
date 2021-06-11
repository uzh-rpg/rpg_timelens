import random
import numpy as np
from collections import defaultdict

from PIL import Image

import torch as th
from timelens.common import event, pytorch_tools, representation
from torch import nn
from torchvision import transforms
import torch



def initialize_transformers(number_of_bins_in_voxel_grid=5):
    return [
        images_to_image_tensors,
        reverse_event_stream_in_before_packet,
        lambda example: event_packets_to_voxel_grids(
            example, number_of_bins_in_voxel_grid
        )
    ]

def event_packets_to_voxel_grids(example, number_of_bins_in_voxel_grid):
    for packet_name in ["before", "after"]:
        example[packet_name]["voxel_grid"] = representation.to_voxel_grid(
            example[packet_name]["events"], number_of_bins_in_voxel_grid
        )
    example["before"]["reversed_voxel_grid"] = representation.to_voxel_grid(
        example["before"]["reversed_events"], number_of_bins_in_voxel_grid
    )
    return example


def apply_transforms(example, transforms):
    if transforms:
        for transform in transforms:
            example = transform(example)
    return example

    
def apply_random_flips(example):
    """Returns example with randomly fliped events and image.
    
    This transformer should be applied before converting events to
    voxel grid.
    """
    # 0 - no flip, 1 - horizontal, 2 - vertical, 3 - both.
    choice = random.randint(0, 3)
    choice_to_axis = {1: 1, 2: 0, 3: (0, 1)}
    if choice == 0:
        return example
    for packet in ["before", "middle", "after"]:
        # note that collor channel is the last.
        image_array = np.array(example[packet]["rgb_image"])
        flipped_image = Image.fromarray(np.flip(image_array, axis=choice_to_axis[choice]))
        example[packet]["rgb_image"] = flipped_image
    for packet in ["before", "after"]:
        event_sequence = example[packet]["events"]
        if choice in [1, 3]:
            event.flip_events_horizontally(event_sequence)
        if choice in [2, 3]:
            event.flip_events_vertically(event_sequence)
    return example


def collate(examples_list):
    """Returns collated examples list."""
    batch = defaultdict(dict)
    batch['middle'] = {}
    for packet_name in ["before", "middle", "after"]:
        if packet_name not in examples_list[0]:
            continue
        for field_name in examples_list[0][packet_name]:
            if ("tensor" in field_name or "voxel_grid" in field_name) and (
                "std" not in field_name and "mean" not in field_name
            ):
                batch[packet_name][field_name] = th.stack(
                    [example[packet_name][field_name] for example in examples_list]
                )
            else:
                batch[packet_name][field_name] = [
                    example[packet_name][field_name] for example in examples_list
                ]
    return batch


def rgb_images_to_gray(example):
    """Converts all rgb PIL images to gray scale and appends them."""
    for packet_name in ["before", "after", "middle"]:
        if (packet_name not in example) or ("rgb_image" not in example[packet_name]):
            continue
        example[packet_name]["gray_image"] = transforms.Grayscale()(
            example[packet_name]["rgb_image"]
        )
    return example


def images_to_image_tensors(example):
    """Converts all PIL images to tensors and appends them."""
    for packet_name in ["before", "after", "middle"]:
        if packet_name not in example:
            continue
        current_fields = list(example[packet_name].keys())
        for field_name in current_fields:
            if "tensor" not in field_name and "image" in field_name:
                image_tensor_field_name = "{}_tensor".format(field_name)
                example[packet_name][image_tensor_field_name] = transforms.ToTensor()(
                    example[packet_name][field_name]
                )
    return example


def reverse_event_stream_in_before_packet(example):
    event_stream = example["before"]["events"].copy()
    event_stream.reverse()
    example["before"]["reversed_events"] = event_stream
    return example

