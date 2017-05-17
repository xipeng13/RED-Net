# Xi Peng, Feb 2017
import os, sys
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from visdom import Visdom
vis = Visdom()

from pylib import FacePts, FaceAug

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
        pts = FacePts.Pts2Lmk(pts_path) # L x 2
    return img, pts
 
class ImageLoader(data.Dataset):
    def __init__( self, list_file, transform=None, is_train=True, 
                  img_shape=[256,256], face_size=200, 
                  resmap_shape=[64,64], heatmap_shape=[128,128] ):
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

        img_aug,pts_aug = FaceAug.AugImgPts(np.array(img), pts, 
                            	self.img_shape[0], self.face_size, 
                            	scale_aug, rotate_aug)

        pts_aug2 = torch.from_numpy(pts_aug).float()
        # debug using visdom
        #img_aug_plot = FacePts.DrawImgPts(img_aug, pts_aug)
        #img_aug_plot = np.asarray(img_aug_plot, dtype='uint8').transpose((2,0,1))
        #vis.image(img_aug_plot, opts=dict(title='img_aug'))
		#exit()

        ### response map for detection
        pts_det = pts_aug * (1.*self.resmap_shape[0]/self.img_shape[0]) # L x 2
        pts_det = FacePts.Lmk68to7(pts_det)
        ann_size = FacePts.CircleSize(base_size=3, scale=scale_aug)

        resmap = FacePts.Lmk2Resmap_mc(pts_det, self.resmap_shape, ann_size)
        wt_resmap = FacePts.GtMap2WeightMap(resmap, reduce_factor=0.5)

        # debug using visdom
        #img_det_plot = img_aug.resize((64,64), Image.ANTIALIAS)
        #img_det_plot = FacePts.DrawImgPts(img_det_plot, pts_det)
        #img_det_plot = np.asarray(img_det_plot, dtype='uint8').transpose((2,0,1))
        #vis.image(img_det_plot, opts=dict(title='img_det'))
        ##vis.scatter(X=pts_det,Y=np.arange(7)+1,opts=dict(title='pts_det'))
        #for c in range(7): 
        #    vis.heatmap(resmap[c,], opts=dict(title='resmap'))
        #    vis.heatmap(wt_resmap[c,], opts=dict(title='wt_resmap'))
        #exit()

        resmap = torch.from_numpy(resmap).float()
        wt_resmap = torch.from_numpy(wt_resmap).float()

        ### heat map for regression
        pts_reg = pts_aug * (1.*self.heatmap_shape[0]/self.img_shape[0]) # L x 2
        heatmap = FacePts.Lmk2Heatmap(pts_reg, self.heatmap_shape, sigma=1)
        heatmap = torch.from_numpy(heatmap).mul(10).float()

        if self.transform_img is not None:
            img_aug = self.transform_img(img_aug) # [0,1], c x h x w

        return img_aug, pts_aug2, resmap, wt_resmap, heatmap

    def __len__(self):
        return len(self.img_list)
