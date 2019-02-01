import predict
from skimage.io import imread
from flask import Flask, request
from converter import *
import zipfile
import os 
import json
import shutil
import dicom
from h import *
import subprocess
from subprocess import call

app = Flask(__name__)

ORIGINAL_KEY = 'original'
TMP_ZIP = './patient.zip'
TMP_DIR = './patient'
OUT_ZIP = 'out.zip'
OUT_DIR = './out'

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if not os.path.isdir(file):
                ziph.write(os.path.join(root, file), file)


@app.route('/CT_LIVER', methods=['POST'])
def CT_SEG():
    ret = run_skeleton(None, "CT_LIVER")
    return ret
@app.route('/MR_LIVER', methods=['POST'])
def MRI_SEG():
    ret = run_skeleton(None, "MR_LIVER")
    return ret
@app.route('/CT_SPLEEN', methods=['POST'])
def CTS_SEG():
    ret = run_skeleton(None, "CT_SPLEEN")
    return ret
@app.route('/CT_PROCESSED', methods=['POST'])
def CT_PROCESSED():
    return p

def run_skeleton(processor_function, seg_type):
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)

    if os.path.exists(OUT_DIR):    
        shutil.rmtree(OUT_DIR)
    os.mkdir(TMP_DIR)
    os.mkdir(OUT_DIR)
    cl = request.content_length

    keys = request.get_json()
    _zip = None
    if ORIGINAL_KEY in keys:
        os.mkdir(os.path.join(TMP_DIR, '1000000'))
        os.mkdir(os.path.join(TMP_DIR, '1000000', ORIGINAL_KEY)) 
     
    _zip = keys[ORIGINAL_KEY]

    Base64ToFile(TMP_ZIP, _zip)
    
    zip_ref = zipfile.ZipFile(TMP_ZIP, 'r') 
    zip_ref.extractall(os.path.join(TMP_DIR, '1000000', ORIGINAL_KEY))
    zip_ref.close()

    P = os.path.join(TMP_DIR, '1000000', ORIGINAL_KEY)

    cvt_dcm_png(P, seg_type)

    if seg_type == "CT_LIVER":
        command = "python3 main.py --mode=infer --infer_data=%s --output_folder=%s --ckpt=/mnt/SSD_BIG2/LSS_LIV/DenseNetCkpt_deeplab_third/-0 --layers_per_block=5,6,6,8,7,9 --batch_size=1 --organ=liver"

        process = subprocess.Popen((command % (P, OUT_DIR)).split(" "))
        process.wait()
    elif seg_type == "CT_SPLEEN":
        command = "python3 main.py --mode=infer --infer_data=%s --output_folder=%s --ckpt=/mnt/SSD_BIG2/LSS_LIV/DenseNetCkpt_deeplab_spleen_first/-1 --layers_per_block=5,6,6,8,7,9 --batch_size=1 --organ=spleen"

        process = subprocess.Popen((command % (P, OUT_DIR)).split(" "))
        process.wait()   
    elif seg_type == "MR_LIVER":
        print("MR_LIVER execute")
        predict.run("patient", "out", "MRI", (512,512)) 


    fs = os.listdir(OUT_DIR)
    fs.sort()
    for idx, f in enumerate(fs):
        write_dicom(imread(os.path.join(OUT_DIR,f)), idx, os.path.join(OUT_DIR,f)[:-4] + ".dcm")
        os.remove(os.path.join(OUT_DIR,f))
        

    zip_out = zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED)
    for f in os.listdir(OUT_DIR):
        zip_out.write(os.path.join(OUT_DIR, f), f)

    zip_out.close()

    encoded = FileToBase64(OUT_ZIP)
    print('encoded length : %s' %(len(encoded)))
    return json.dumps({ORIGINAL_KEY:encoded}) 

if __name__ == '__main__':
    app.run(host='0.0.0.0')
