import os

import numpy as np
from mpl_toolkits import axes_grid1
from PIL import Image
import cv2

import matplotlib  # noqa #isort:skip

matplotlib.use("Agg")  # isort:skip
import matplotlib.colors as mcolors  # isort:skip
import matplotlib.patches as mpatches  # isort:skip
from matplotlib import pyplot as plt  # isort:skip

    
def _make_palette(colors):
    palette = []
    for color in colors:
        palette.append(mcolors.to_rgb(color))
    return np.array(palette)


def save_index_matrix(filename, index_matrix, index_to_color, index_to_name):
    """Show matrix with indices.
    
    Args:
        filename: file where figure will be saved.
        index_matrix: 2d matrix with indices.
        index_to_color: list with colornames. E.g. if it is equal to
                        ['green', 'red', 'blue'], when locations with 
                        index 0, will be shown in green.
                        
        index_to_name: list of index names. E.g. if it is equal to 
                       ['first', 'second', 'third'], when figure will 
                       have colorbar as follows:
                       green - first
                       red - second
                       blue - fird.
    """
    palette = _make_palette(index_to_color)
    color_matrix = palette[index_matrix]
    figure = plt.figure()
    plot = plt.imshow(color_matrix)
    patches = [
        mpatches.Patch(color=color, label=name)
        for name, color in zip(index_to_name, index_to_color)
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()


class Logger(object):
    """Object for logging training progress."""

    def __init__(self, filename):
        self._filename = filename

    def log(self, text):
        """Appends text line to the file."""
        if os.path.isfile(self._filename):
            handler = open(self._filename, "r")
            lines = handler.readlines()
            handler.close()
        else:
            lines = []
        lines.append(text + "\n")
        handler = open(self._filename, "w")
        handler.writelines(lines)
        handler.close()


def _add_scaled_colorbar(plot, aspect=20, pad_fraction=0.5, **kwargs):
    """Adds scaled colorbar to existing plot."""
    divider = axes_grid1.make_axes_locatable(plot.axes)
    width = axes_grid1.axes_size.AxesY(plot.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_axis = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_axis)
    return plot.axes.figure.colorbar(plot, cax=cax, **kwargs)


def make_blinking_images_video(
    filename, image_list, time_per_image=0.2, number_of_loops=10
):
    """Creates video that flips between images in the list.
    
    Args:
        time_per_image: is time interval in seconds during which each image is 
                        shown.
        image_list: list of PIL image. 
    """
    height, width = image_list[0].shape
    video_fps = 30
    number_of_frames_per_image = round(video_fps * time_per_image)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(filename, fourcc, video_fps, (width, height))
    for _ in range(number_of_loops):
        for image in image_list:
            if image.shape[:2] != (height, width):
                raise ValueError("Not all images have the same size.")
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for _ in range(number_of_frames_per_image):
                video.write(bgr_image)
    video.release()


def save_image(filename, image):
    """Save color image to file.
    Args:
        filename: image file where the image will be saved..
        image: 3d image tensor.
    """
    figure = plt.figure()
    numpy_image = image.permute(1, 2, 0).numpy()
    plot = plt.imshow(numpy_image.astype(np.uint8))
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()


def save_matrix(
    filename,
    matrix,
    minimum_value=None,
    maximum_value=None,
    colormap="magma",
    is_colorbar=True,
):
    """Saves the matrix to the image file.
    Args:
        filename: image file where the matrix will be saved.
        matrix: tensor of size (height x width). Some values might be
                equal to inf.
        minimum_value, maximum value: boundaries of the range.
                                      Values outside ot the range are
                                      shown in white. The colors of other
                                      values are determined by the colormap.
                                      If maximum and minimum values are not
                                      given they are calculated as 0.001 and
                                      0.999 quantile.
        colormap: map that determines color coding of matrix values.
    """
    figure = plt.figure()
    noninf_mask = matrix != float("inf")
    if minimum_value is None:
        minimum_value = np.quantile(matrix[noninf_mask], 0.001)
    if maximum_value is None:
        maximum_value = np.quantile(matrix[noninf_mask], 0.999)
    plot = plt.imshow(matrix.numpy(), colormap, vmin=minimum_value, vmax=maximum_value)
    if is_colorbar:
        _add_scaled_colorbar(plot)
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()


def _plot_on_axis(
    axis,
    data,
    legend_template,
    y_axis_label,
    color_of_axis_and_plot,
    linestyle,
    marker,
    is_error=True,
):
    opt_value, opt_index = np.max(data), np.argmax(data) + 1
    if is_error:
        opt_value, opt_index = np.min(data), np.argmin(data) + 1
    epochs = range(1, len(data) + 1)
    legend = legend_template.format(opt_value, opt_index)
    plot_handel = axis.plot(
        epochs,
        data,
        linestyle=linestyle,
        marker=marker,
        color=color_of_axis_and_plot,
        label=legend,
    )[0]
    axis.set_ylabel(y_axis_label, color=color_of_axis_and_plot)
    axis.set_xlabel("Epoch")
    return plot_handel


def plot_with_two_y_axis(
    filename,
    left_plot_data,
    right_plot_data,
    left_plot_legend_template="Training loss (smallest {0:.3f}, epoch {1:})",
    right_plot_legend_template="Validation error (smallest {0:.3f}, epoch {1:})",
    right_y_axis_label="Validation error, [%]",
    left_y_axis_label="Training loss",
    left_is_error=True,
    right_is_error=True,
):
    """Plots two graphs on same figure.
    
    The figure has two y-axis the left and the right which correspond
    to two plots. The axis have different scales. The left axis and
    the corresponding plot are shown in blue and the right axis and
    the corresponding plot are shown in red.
    
    Args:
        filename: image file where plot is saved.
        xxxx_plot_data: list with datapoints. Every element of the
                        list corresponds to an epoch.
        xxxx_plot_legend_template: template for the plot legend.
        xxxx_y_axis_label: label of the axis.
        xxxx_is_error: if true than show minimum value and corresponding 
                       argument, if false, show maximum value.
    """
    figure, left_axis = plt.subplots()
    left_plot_handle = _plot_on_axis(
        left_axis,
        left_plot_data,
        legend_template=left_plot_legend_template,
        y_axis_label=left_y_axis_label,
        color_of_axis_and_plot="blue",
        linestyle="dashed",
        marker="o",
        is_error=left_is_error,
    )
    right_axis = left_axis.twinx()
    right_plot_handle = _plot_on_axis(
        right_axis,
        right_plot_data,
        legend_template=right_plot_legend_template,
        y_axis_label=right_y_axis_label,
        color_of_axis_and_plot="red",
        linestyle="solid",
        marker="o",
        is_error=right_is_error,
    )
    right_axis.legend(handles=[left_plot_handle, right_plot_handle])
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
    return figure, left_axis, right_axis


def plot_losses_and_errors(filename, losses, errors):
    plot_with_two_y_axis(filename, losses, errors)


def plot_points_on_background(y, x, background, points_color=[0, 0, 255]):
    """Return PIL image with overlayed points.
    Args:
        x, y : numpy vectors with points coordinates (might be empty).
        background: (height x width x 3) torch tensor.
        color: color of points [red, green, blue] uint8.
    """
    if x.size == 0:
        return background
    background = np.array(background)
    if not (len(background.shape) == 3 and background.shape[-1] == 3):
        raise ValueError("background should be (height x width x color).")
    height, width, _ = background.shape
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
        raise ValueError('points coordinates are outsize of "background" ' "boundries.")
    background[y, x, :] = points_color
    return Image.fromarray(background)
