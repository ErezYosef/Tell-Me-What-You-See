
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import random
# import numpy as np
import torch
import cv2
from glob import glob
from natsort import natsorted
import os
from PIL import Image
import yaml

from .dataset_s7_cam import Dataset_s21

class Dataset_allied_cam(Dataset_s21):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.amplitude_factors = {'iso50_t1-50': 1,
                                  'iso50_t1-45': 0.9,
                                  'iso3200_t1-12000': 3.75,
                                  'iso3200_t1-6000': 1.875,
                                  'to_coco_mean': 110,  # 122
                                  }
        white_balance_to_coco_mean = torch.tensor([10.6896, 20.5843, 20.5710, 10.9262]).view(4, 1, 1)

        self.amplitude_factors = {'t7000gain0': 1,
                                  't700gain20': 1, # 10^{\frac{g}{20}}\cdot\frac{t}{7000} desmos
                                  't350gain26.02': 1,
                                  't175gain32.04': 1,
                                  't175gain26.02': 2,
                                  't140gain27.96': 2,
                                  't100gain30.88': 2,
                                  'to_coco_mean': white_balance_to_coco_mean,  # 122
                                  }

        self.files = [f for f in self.files if f.endswith('.pt')]
        self.files.sort()
        print('sorted and filtered files:', len(self.files))
        print(f'Warning: using {kwargs["fpath_gt"]} and {kwargs["fpath_noisy"]} from {kwargs["main_path_dataset"]}')


