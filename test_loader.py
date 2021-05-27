import argparse
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt


def render(events, image):
    H, W, _ = image.shape
    x = events['x'].astype("int")
    y = events['y'].astype("int")
    p = events['p']

    mask = (x <= W-1) & (y <= H-1) & (x >= 0) & (y >= 0)
    x_ = x[mask]
    y_ = y[mask]
    p_ = p[mask]

    image[y_,x_,:] = 0
    image[y_,x_,p_] = 255
   
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Example Loader for a sample")
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--dataset_type", default="close")
    parser.add_argument("--sequence", default="fountain_schaffhauserplatz_02")
    parser.add_argument("--sample_index", type=int, default=200)

    args = parser.parse_args()
    event_file = join(args.dataset_root, "%s/test/%s/events_aligned/%06d.npz" % (args.dataset_type, args.sequence, args.sample_index))
    image_file = join(args.dataset_root, "%s/test/%s/images_corrected/%06d.png" % (args.dataset_type, args.sequence, args.sample_index))

    image = cv2.imread(image_file)[...,::-1]
    events = np.load(event_file)

    rendered_image = render(events, image)
    
    plt.imshow(rendered_image)
    plt.show()
    
    
