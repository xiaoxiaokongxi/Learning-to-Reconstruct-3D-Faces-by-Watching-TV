import matplotlib.pyplot as plt
import face_alignment
from skimage import io
import numpy as np
import os

#load image folder file path
imgfolder = 'G:\paper\Friends'
kptfolder = 'G:\paper\Friends\kpt'

# segfolder = 'G:\paper\Friends\seg'

filename = os.listdir(imgfolder)
for folder in filename:
    #get all name in filenames
    path = os.path.abspath(os.path.join(imgfolder, folder))  # ../imagefolder/id_x

    #check is image folder
    if 'id_' in folder:
        print('processing folder {}'.format(folder))
        # create folder for each id in kpt and seg folder dirs
        if os.path.exists(os.path.join(kptfolder, folder)) == False:
            os.makedirs(os.path.join(kptfolder, folder))
            print('{} created'.format(os.path.join(kptfolder, folder)))

        #if used for seg
        # if os.path.exists(os.path.join(segfolder, folder)) == False:
        #     os.makedirs(os.path.join(segfolder, folder))
        #     print('{} created'.format(os.path.join(segfolder, folder)))
        
        # make prediction for kpt and save
        fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        for imgfile in os.listdir(path):
            image = io.imread(os.path.join(path,imgfile))
            preds = fan.get_landmarks(image)
            # print(preds)
            if preds is not None:
                np.savetxt(os.path.join(os.path.join(kptfolder, folder), imgfile[:-4] + '_kpt2d.txt'), preds[0])
                print(os.path.join(os.path.join(kptfolder, folder), imgfile[:-4] + '_kpt2d.txt' + ' created'))


