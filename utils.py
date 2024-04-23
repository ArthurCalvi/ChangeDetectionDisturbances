
import os
from tqdm import tqdm
import rasterio 
import numpy as np 

def load_folder(folder, func=None, func_args=None):
    files = os.listdir(folder)
    files = [f for f in files if (f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png'))]
    files.sort()
    data = []
    for file in tqdm(files):
        with rasterio.open(os.path.join(folder, file)) as src:
            mask = src.read().squeeze()
            if func is not None:
                mask = func(mask, **func_args)
            data.append(mask)
    return np.array(data)


def normalize(array, gain=2):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return gain * ((array - array_min)/(array_max - array_min))