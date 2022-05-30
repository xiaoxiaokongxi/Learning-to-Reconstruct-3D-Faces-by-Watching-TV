import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pdb

class VGGFace2Dataset(Dataset):
    def __init__(self, K, image_size, scale, trans_scale = 0, isTemporal=False, isEval=False, isSingle=False):
        '''
        K must be less than 6
        '''
        self.K = K
        self.image_size = image_size
        self.imagefolder = 'data/new/id1/jpg'
        self.kptfolder = 'data/new/id1/kpt'
        self.segfolder = 'data/new/id1/seg/masks'
        # hq:
        # datafile = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_bbx_size_bigger_than_400_train_list_max_normal_100_ring_5_1_serial.npy'
        # datafile = '/content/drive/MyDrive/Colab_Notebooks/Friends/serial.npy'
        # if isEval:
        #     datafile = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_val_list_max_normal_100_ring_5_1_serial.npy'
        # self.data_lines = np.load(datafile).astype('str')

        self.isTemporal = isTemporal
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

        files = os.listdir(self.imagefolder)
        files.sort()
        self.name_files = files

    def __len__(self):
        return len(self.name_files) // 10

    def __getitem__(self, idx):
        images_list = []; kpt_list = []; mask_list = []

        init_k = np.random.randint(10-self.K)
        names = self.name_files[idx*10+init_k: idx*10+init_k+self.K]

        for name in names:
            name = name[:-4]
            image_path = os.path.join(self.imagefolder, name + '.jpg')
            seg_path = os.path.join(self.segfolder, name + '.npy')
            kpt_path = os.path.join(self.kptfolder, name + '_kpt2d.npy')

            image = imread(image_path)/255.
            kpt = np.load(kpt_path)[:,:2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            images_list.append(cropped_image.transpose(2,0,1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32) #K,224,224,3
        # print(images_array.shape, kpt_array.shape, mask_array.shape)
        # if self.isSingle:
        #     images_array = images_array.squeeze()
        #     kpt_array = kpt_array.squeeze()
        #     mask_array = mask_array.squeeze()

        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'mask': mask_array
        }

        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]);
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask



class VGGFace2HQDataset(Dataset):
    def __init__(self, K, image_size, scale, trans_scale = 0, isTemporal=False, isEval=False, isSingle=False):
        '''
        K must be less than 6
        '''
        self.K = K
        self.image_size = image_size
        self.imagefolder = '/content/drive/MyDrive/Colab_Notebooks/Friends/id_2'
        self.kptfolder = '/content/drive/MyDrive/Colab_Notebooks/Friends/kpt/id_2'
        self.segfolder = '/content/drive/MyDrive/Colab_Notebooks/Friends/seg/id_2/masks'
        # hq:
        # datafile = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_bbx_size_bigger_than_400_train_list_max_normal_100_ring_5_1_serial.npy'
        datafile = '/content/drive/MyDrive/Colab_Notebooks/Friends/serial.npy'
        self.data_lines = np.load(datafile).astype('str')

        self.isTemporal = isTemporal
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        images_list = []; kpt_list = []; mask_list = []

        name = self.data_lines[idx][:-4]
        image_path = os.path.join(self.imagefolder, name + '.jpg')
        seg_path = os.path.join(self.segfolder, name + '.npy')
        kpt_path = os.path.join(self.kptfolder, name + '_kpt2d.npy')

        image = imread(image_path)/255.
        kpt = np.load(kpt_path)[:,:2]
        mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

        ### crop information
        tform = self.crop(image, kpt)
        ## crop
        cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

        # normalized kpt
        cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

        images_list.append(cropped_image.transpose(2,0,1))
        kpt_list.append(cropped_kpt)
        mask_list.append(cropped_mask)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32) #K,224,224,3

        if self.isSingle:
            images_array = images_array.squeeze()
            kpt_array = kpt_array.squeeze()
            mask_array = mask_array.squeeze()

        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'mask': mask_array
        }

        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]);
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask
