import errno
import matplotlib

matplotlib.use('Agg')
from joblib import Parallel, delayed
import os, fnmatch
from skimage.io import imread, imsave
import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset
import numpy as np
import datetime, time

def proc_mr(f):
   
    ds = dicom.read_file(f, force=True)
    ds_h = ds

    ds = ds.pixel_array.astype(np.int16)

    imsave(f[:-4] + ".png", ds.astype(np.uint16))

def proc_ct(f):
 
    ds = dicom.read_file(f, force=True)
    ds_h = ds
    inum = 0
    try:
        inum = ds[0x20,0x13].value
    except:
        pass
    ds = ds.pixel_array.astype(np.int16) - 1024
    ds[ds < 0] = 0
    ds[ds > 400] = 0

    imsave(os.path.dirname(f[:-4]) +"/"+ str(inum).zfill(10) + os.path.basename(f[:-4]) + ".png", ds.astype(np.uint16))

def cvt_dcm_png(directory, seg_type):
    for k in [os.path.join(directory,d) for d in os.listdir(directory)]:
        print(k[-3:])
        if k[-3:] == "png":
            break
        if "CT" in seg_type:
            
            proc_ct(k)
        elif seg_type == "MR_LIVER":
            proc_mr(k)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def write_dicom(pixel_array, index, filename):
    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
    target_dir = '/'.join(filename.split('/')[:-1]) 
    print(target_dir)
    mkdir_p(target_dir)
    file_meta = Dataset()
    ds = FileDataset(filename, {},file_meta = file_meta)
    ds.ContentDate = str(datetime.date.today()).replace('-','')
    ds.ContentTime = str(time.time()) #milliseconds since the epoch

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.InstanceNumber = index
    #ds.PhotometricInterpretation = bytes("MONOCHROME2", "UTF-8")
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    #ds.SmallestImagePixelValue = bytes('\\x00\\x00', 'UTF-8')
    #ds.LargestImagePixelValue = bytes('\\xff\\xff', 'UTF-8')
            
    ds.Columns = pixel_array.shape[1]
    ds.Rows = pixel_array.shape[0]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()
    ds.save_as(filename)
    #Parallel(n_jobs=60)(delayed(proc))
