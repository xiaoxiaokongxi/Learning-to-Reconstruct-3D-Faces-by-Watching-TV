import matplotlib.pyplot as plt
import face_alignment
from skimage import io
import os

#load image folder file path
imgfolder = 'G:\paper\Friends'
kptfolder = 'G:\paper\Friends\kpt'
segfolder = 'G:\paper\Friends\seg'

filename = os.listdir(imgfolder)
for folder in filename:
    #get all name in filenames
    path = os.path.abspath(os.path.join(imgfolder, folder))

    #check is image folder
    if 'id_' in folder:
        print('processing folder {}'.format(folder))
        # create folder for each id in kpt and seg folder dirs
        if os.path.exists(os.path.join(kptfolder, folder)) == False:
            os.makedirs(os.path.join(kptfolder, folder))
            print('{} created'.format(os.path.join(kptfolder, folder)))
        if os.path.exists(os.path.join(segfolder, folder)) == False:
            os.makedirs(os.path.join(segfolder, folder))
            print('{} created'.format(os.path.join(segfolder, folder)))
        
        # make prediction for kpt and save
        fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        for imgfile in os.listdir(path):
            image = io.imread(os.path.join(path,imgfile))
            preds = fan.get_landmarks(image)
            # refer to DECA
            # landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
            # np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())




