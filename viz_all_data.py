"""
visualize all data
"""
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from timelens.common.hybrid_storage import HybridStorage


def render(events, image):
    x,y,p = events[:,0].astype(np.int32), events[:,1].astype(np.int32), events[:,3]>0
    height, width = image.shape[:2]
    x = np.clip(x, 0, width-1)
    y = np.clip(y, 0, height-1)
    image[y[p],x[p],1] = 255
    image[y[p==0],x[p==0],0] = 255
    return image


def viz_video(folder, number_of_skips, viz):
    event_folder = os.path.join(folder, 'events_aligned')
    image_folder = os.path.join(folder, 'images_corrected')
    hybrid = HybridStorage.from_folders_jit(
            event_folder,
            image_folder,
            event_file_template="*.npz",
            image_file_template="*.png",
            cropping_data=None,
            timestamps_file="timestamp.txt"
    )

    it1 = hybrid.make_interframe_events_iterator(number_of_skips)
    it2 = hybrid.make_pair_boundary_timestamps_iterator(number_of_skips)
    it3 = hybrid.make_boundary_frames_iterator(number_of_skips)
    total = len(hybrid._images._timestamps)-1
    for events, (left_ts, right_ts), (left_img, right_img) in tqdm(zip(it1,it2,it3), total=total):
        #check events are in [left_ts, right_ts]
        start_ts = events._start_time
        end_ts = events._end_time
        assert events._features[0,2] >= start_ts
        assert events._features[-1,2] <= end_ts
        assert events._features[0,2] >= left_ts
        assert events._features[-1,2] <= right_ts

        assert start_ts == left_ts
        assert end_ts == right_ts

        if viz:
            left_img, right_img = np.array(left_img), np.array(right_img)
            left_events, right_events = events.split_in_two((start_ts+end_ts)/2)
            left_img = render(left_events._features, left_img)
            right_img = render(right_events._features, right_img)
            cat = np.concatenate((left_img, right_img), axis=1)
            cv2.imshow('data', cat[...,::-1]) #opencv is bgr
            key = cv2.waitKey(5)
            if key == 27:
                break



def viz_dataset_type(path, number_of_skips, viz):
    folders = os.listdir(path)
    files = [os.path.join(path, item) for item in folders]
    for folder in files:
        viz_video(folder, number_of_skips, viz)


def main(path, number_of_skips=1, viz=False):
    for dataset_type in ['close', 'far']:
        viz_dataset_type(os.path.join(path, dataset_type, 'test'), number_of_skips, viz)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visualize All of Timelens")
    parser.add_argument("path", default="")
    parser.add_argument("--number_of_skips", type=int, default=1)
    parser.add_argument("--viz", type=bool, default=True)
    args = parser.parse_args()
    main(args.path, args.number_of_skips, args.viz)
