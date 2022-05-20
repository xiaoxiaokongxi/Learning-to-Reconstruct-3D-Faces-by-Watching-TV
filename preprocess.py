import matplotlib.pyplot as plt
# import face_alignment
from skimage import io
import numpy as np
import os
from os import sys

sys.path.append('.')
sys.path.append('./face-parsing.PyTorch')
from pred_mask import evaluate

#load image folder file path
imgfolder = '/Users/xiyichen/Documents/3d_vision/Learning-to-Reconstruct-3D-Faces-by-Watching-TV'
kptfolder = '/Users/xiyichen/Documents/3d_vision/kpts'
segfolder = '/Users/xiyichen/Documents/3d_vision/segs'
face_segmentation_model_path = '/Users/xiyichen/Documents/3d_vision/Learning-to-Reconstruct-3D-Faces-by-Watching-TV/face-parsing.PyTorch/res/cp/79999_iter.pth'

filename = os.listdir(imgfolder)
for folder in filename:
    #get all name in filenames
    path = os.path.abspath(os.path.join(imgfolder, folder))  # ../imagefolder/id_x

    #check is image folder
    if 'id_' in folder:
        print('processing folder {}'.format(folder))

        id_segfolder = os.path.join(segfolder, folder)
        if os.path.exists(id_segfolder) == False:
            os.makedirs(id_segfolder)
            print('{} created'.format(id_segfolder))

        masks = evaluate(respth=id_segfolder,
                         dspth=folder,
                         model_path=face_segmentation_model_path,
                         save_masks=True,
                         save_imgs=False)


        # create folder for each id in kpt and seg folder dirs
        id_kptfolder = os.path.join(kptfolder, folder)
        if os.path.exists(id_kptfolder) == False:
            os.makedirs(id_kptfolder)
            print('{} created'.format(id_kptfolder))

        # if used for seg
        # make prediction for kpt and save
        fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        for imgfile in os.listdir(path):
            image = io.imread(os.path.join(path, imgfile))
            preds = fan.get_landmarks(image)
            # print(preds)
            if preds is not None:
                np.savetxt(os.path.join(os.path.join(kptfolder, folder), imgfile[:-4] + '_kpt2d.txt'), preds[0])
                print(os.path.join(os.path.join(kptfolder, folder), imgfile[:-4] + '_kpt2d.txt' + ' created'))

