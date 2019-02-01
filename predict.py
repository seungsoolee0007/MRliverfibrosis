import errno
import os
import matplotlib
matplotlib.use("Agg")
import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset
import numpy as np
import datetime, time
from scipy.ndimage.morphology import binary_fill_holes
import scipy.misc
import tensorflow as tf
import random
import SegNetCMR as sn
import numpy as np
from skimage.transform import resize
from skimage import img_as_uint
from skimage.io import imsave
import dicom
import scipy.ndimage
from skimage import measure, morphology
import png
# visualization library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource 
import cv2

HAVE_GPU = False
SAVE_INTERVAL = 2

PREDICT_DIR = '/mnt/SSD_BIG2/LSS_LIV/MLbackup/Remote_OrganSegNet512/patient'
OUTPUT_DIR = '/mnt/SSD_BIG2/LSS_LIV/MLbackup/Remote_OrganSegNet512/out'
RUN_NAME_CT1 = "Run3x3_CT_LIVER"
RUN_NAME_CT2 = "Run3x3_CT_MAX"

RUN_NAME_MRI = "Run3x3_HM_LIVER"
CONV_SIZE = 3

ROOT_LOG_DIR = './Output'
CHECKPOINT_FN = 'model.ckpt'

#Start off at 0.9, then increase.
BATCH_NORM_DECAY_RATE = 0.9

MAX_STEPS = 20000
BATCH_SIZE = 1 

LOG_DIR_CT1 = os.path.join(ROOT_LOG_DIR, RUN_NAME_CT1)
LOG_DIR_CT2 = os.path.join(ROOT_LOG_DIR, RUN_NAME_CT2)
LOG_DIR_MRI = '/mnt/SSD_BIG2/LSS_LIV/MLbackup/Remote_OrganSegNet512/Output/Run3x3_HM_LIVER'



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def run(predict_dir, output_dir, seg_type, dimension):
    import SegNetCMR as sn
    predicting_data = sn.GetData(predict_dir, seg_type)
   
    LOG_DIR = LOG_DIR_CT2
    if seg_type == "MRI":
        LOG_DIR = LOG_DIR_MRI
    elif seg_type == "CT2":
        LOG_DIR=LOG_DIR_CT2
    g = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    os.makedirs(output_dir, exist_ok=True)
    with g.as_default():
        images, labels, is_training = sn.placeholder_inputs(batch_size=BATCH_SIZE)
        logits = sn.inference(images=images, is_training=is_training, conv_size=CONV_SIZE, batch_norm_decay_rate=BATCH_NORM_DECAY_RATE, have_gpu=HAVE_GPU)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name])   
        for patient_dir, scan, pimages in predicting_data.getPatient():
            tmp_image = None

            sm = tf.train.SessionManager()
            sess = sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR, config=config)
            sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))
            for index, image in enumerate(pimages):



                print(index, patient_dir[index])
                train_feed_dict = {images: np.reshape(cv2.resize(image, (256,256)), (BATCH_SIZE, 256, 256, 1)), is_training: False}
                logits_cal =  sess.run([tf.argmax(logits, 3)], feed_dict=train_feed_dict)
                print(scan[index])
                tmp_image = cv2.resize(np.reshape(logits_cal[0][0].astype(np.float64),(256,256)), (scan[index][1], scan[index][0])).astype(np.bool)


                tmp_image = morphology.remove_small_objects(tmp_image, 3300, connectivity=4)
                tmp_image = binary_fill_holes(tmp_image).astype(np.bool)
                #write_dicom(tmp_image, index, os.path.join(OUTPUT_DIR, str(index) + '.dcm'))

                #mkdir_p(os.path.join(output_dir, os.path.dirname(patient_dir[index])))
                print("patient dir", os.path.join(output_dir, os.path.dirname(patient_dir[index])))
                imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(patient_dir[index]))[0] + '_mask.png'), tmp_image * 255)
            

        sess.close()


def resample(image, scan, new_spacing=[3,2,2]):

    # Determine current pixel spacing
    spacing = np.array([scan.SliceThickness] + scan.PixelSpacing, dtype=np.float32)
    print("image size: " + str(image.shape), scan.SliceThickness, scan.PixelSpacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    print("interpolation start")    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='wrap')
    print("interpolation end")
    return image, new_spacing

def plot_3d(image, threshold=0):
   
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    print("marching cube start")
    verts, faces = measure.marching_cubes(p, threshold)
    print("marching cubeend")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    print("figure showed")
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.220)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

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
    return


