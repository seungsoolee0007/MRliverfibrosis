import matplotlib

matplotlib.use('Agg')
from joblib import Parallel, delayed
import os, fnmatch
from skimage.io import imread, imsave
import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset
import numpy as np
import datetime, time


#for dirs in os.listdir(ROOT):
def proc(f):

    ds = dicom.read_file(f, force=True)
    ds_h = ds

    ds = ds.pixel_array.astype(np.int16) - 1024
    ds[ds < 0] = 0
    ds[ds > 400] = 0
    imsave(f[:-4] + ".png", ds.astype(np.uint16))

