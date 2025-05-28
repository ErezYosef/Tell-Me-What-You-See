
from . import dataset_s7_cam, dataset_allied_cam




def get_dataset(name):
    if name == 'sidd':
        return dataset_sidd.Dataset_PairedImage_crops_less_clip
    elif name == 's21':
        return dataset_s7_cam.Dataset_s21
    elif name == 's21_set_caption':
        return dataset_s7_cam.Dataset_s21_set_caption
    elif name == 'allied_cam':
        return dataset_allied_cam.Dataset_allied_cam
    else:
        print(f'Warning: dataset class {name} is missing > return None..')
        return None
