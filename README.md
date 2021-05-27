# High Speed Event and RGB (HS-ERGB) dataset

<img src="assets/hsergb_preview.gif" width="700">

This repository is about the High Speed Event and RGB (HS-ERGB) dataset, used in the 2021 CVPR paper [**TimeLens: Event-based Video Frame Interpolation**](http://rpg.ifi.uzh.ch/docs/CVPR21_Gehrig.pdf) by Stepan Tulyakov*, [Daniel Gehrig*](https://danielgehrig18.github.io/), Stamatios Georgoulis, Julius Erbach, Mathias Gehrig, Yuanyou Li, and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html).

For more information, visit our [project page](http://rpg.ifi.uzh.ch/timelens).

### Citation
A pdf of the paper is [available here](http://rpg.ifi.uzh.ch/docs/CVPR21_Gehrig.pdf). If you use this dataset, please cite this publication as follows:

```bibtex
@Article{Tulyakov21CVPR,
  author        = {Stepan Tulyakov and Daniel Gehrig and Stamatios Georgoulis and Julius Erbach and Mathias Gehrig and Yuanyou Li and
                  Davide Scaramuzza},
  title         = {{TimeLens}: Event-based Video Frame Interpolation},
  journal       = "IEEE Conference on Computer Vision and Pattern Recognition",
  year          = 2021,
}
```

### Download
Download the dataset from our [project page](http://rpg.ifi.uzh.ch/timelens).

### Dataset Structure
The dataset structure is as follows

```
.
├── close
│   └── test
│       ├── baloon_popping
│       │   ├── events_aligned
│       │   └── images_corrected
│       ├── candle
│       │   ├── events_aligned
│       │   └── images_corrected
│       ...
│
└── far
    └── test
        ├── bridge_lake_01
        │   ├── events_aligned
        │   └── images_corrected
        ├── bridge_lake_03
        │   ├── events_aligned
        │   └── images_corrected
        ...

```
Each `events_aligned` folder contains events in the form of several files with template filename `%06d.npz`, and `images_corrected` contains images in the form of several files with template filename `%06d.png`. The `events_aligned` each event file with index `n` contains events between images with index `n-1` and `n`, i.e. event file `000001.npz` contains events between images `000000.png` and `000001.png`. Moreover, `images_corrected` also contains `timestamp.txt` where image timestamps are stored. Note that some folders may contain too many images, however, the number of image stamps in `timestamp.txt` should match.

For a quick test on loading the dataset download the dataset to a folder using the link sent by email.

  wget download_link.zip -O /tmp/dataset.zip
  cd /tmp
  unzip /tmp/dataset.zip
  cd hsergb/
  
Then download this repo

  git clone git@github.com:uzh-rpg/rpg_hs_ergb_dataset.git
  
And run the test

  python rpg_hs_ergb_dataset/test_loader.py --dataset_root . \ 
                                            --dataset_type close \ 
                                            --sequence spinning_umbrella \ 
                                            --sample_index 400
                        

  
