# Xi Peng, Feb 2017
import os, sys
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PXPts, PXAugFace, util


def default_loader(path):
    return Image.open(path).convert('RGB')

def read_img_pts(img_pts_path):
    token = img_pts_path.split(' ')
    # load image
    img_path = token[0]
    img = default_loader(img_path)
    # load pts
    pts_path = token[1]
    if pts_path[-4:] == '.txt':
        pts = np.loadtxt(pts_path) # L x 2
    elif pts_path[-4:] == '.pts':
        pts = PXPts.Pts2Lmk(pts_path) # L x 2
    return img, pts
 
class ImageList(data.Dataset):
    def __init__( self, list_file, transform=None, is_train=True, 
                  img_shape=[256,256], face_size=200, 
                  resmap_shape=[128,128], heatmap_shape=[128,128] ):
        img_list = [line.rstrip('\n') for line in open(list_file)]
        print('total %d images' % len(img_list))

        self.img_list = img_list
        self.transform = transform
        self.is_train = is_train
        self.img_shape = img_shape
        self.face_size = face_size
        self.resmap_shape = resmap_shape
        self.heatmap_shape = heatmap_shape
        self.transform_img = transforms.Compose([self.transform])

    def __getitem__(self, index):
        img, pts = read_img_pts( self.img_list[index] )
        
        SCALE_VAR, ROTATE_VAR = 0.3, 30
        if self.is_train:
            scale_aug = np.random.uniform(1-SCALE_VAR, 1+SCALE_VAR-0.1, 1)
            rotate_aug = np.random.uniform(-ROTATE_VAR, ROTATE_VAR, 1)
        else:     
            scale_aug = np.random.uniform(1, 1, 1)
            rotate_aug = np.random.uniform(0, 0, 1)

        img_aug,pts_aug = PXAugFace.AugImgPts(np.array(img), pts, 
                            self.img_shape[0], self.face_size, 
                            scale_aug, rotate_aug)
        pts_aug /= (self.img_shape[0]/self.heatmap_shape[0]) # L x 2

        ## response map for detection
        pts7 = PXPts.Lmk68to7(pts_aug)
        ann_size = PXPts.CircleSize(base_size=4, scale=scale_aug)
        resmap = PXPts.Lmk2Resmap(pts7, self.resmap_shape, ann_size) # w x h
        resmap = util.GrayPILImageToTensor255(resmap).long() # [0,L), h x w

        ## heat map for regression
        heatmap = PXPts.Lmk2Heatmap(pts_aug, self.heatmap_shape, sigma=1)
        heatmap = torch.from_numpy(heatmap).mul(100).float()

        ## pts
        pts7 = torch.from_numpy(pts7)
        pts = torch.from_numpy(pts_aug)

        if self.transform_img is not None:
            img_aug = self.transform_img(img_aug) # [0,1], c x h x w

        return img_aug, resmap, heatmap, pts7, pts

    def __len__(self):
        return len(self.img_list)