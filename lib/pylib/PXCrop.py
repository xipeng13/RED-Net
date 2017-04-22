# Xi Peng, July 18 2016
import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PXIO, PXPts

def Lmk2Bbox(lmk, ratio):
    # lmk: NLMK x 2  numpy.matrix.shape: (NLMK,2)
    NLMK,dim = lmk.shape
    if dim != 2:
        lmk = lmk.T
    ptx = np.squeeze( np.asarray(lmk[:,0]) )
    pty = np.squeeze( np.asarray(lmk[:,1]) )
    ptx = ptx[np.where(ptx>0)]
    pty = pty[np.where(pty>0)]

    centerx = (min(ptx)+max(ptx)) / 2
    centery = (min(pty)+max(pty)) / 2
    radius = max(max(ptx)-min(ptx), max(pty)-min(pty))
    sl = np.sqrt( 2 * radius**2 ) * ratio
    bbox = [centerx-sl/2, centerx+sl/2, centery-sl/2, centery+sl/2] # l,r,b,t
    bbox = np.round(bbox).astype(int)
    return bbox

def Lmk2BboxTight(lmk):
    # lmk: NLMK x 2  numpy.matrix.shape: (NLMK,2)
    NLMK,dim = lmk.shape
    if dim != 2:
        lmk = lmk.T
    ptx = np.squeeze( np.asarray(lmk[:,0]) )
    pty = np.squeeze( np.asarray(lmk[:,1]) )
    ptx = ptx[np.where(ptx>0)]
    pty = pty[np.where(pty>0)]

    centerx = (min(ptx)+max(ptx)) / 2
    centery = (min(pty)+max(pty)) / 2
    sl = max(max(ptx)-min(ptx), max(pty)-min(pty))
    bbox = [centerx-sl/2, centerx+sl/2, centery-sl/2, centery+sl/2] # l,r,b,t
    bbox = np.round(bbox).astype(int)
    return bbox

def CropImgPtsPreserveResolution(img, lmk):
    rows,cols,chs = img.shape
    nlmk,dim = lmk.shape
    if dim != 2:
        lmk = lmk.T
        nlmk,dim = lmk.shape
    if nlmk==68:
        ratio = 3
    elif nlmk==7:
        ratio = 5
    bbox = Lmk2BboxTight(lmk, ratio)
    cx = (bbox[0]+bbox[1]) / 2.0
    cy = (bbox[2]+bbox[3]) / 2.0
    sl = bbox[1] - bbox[0]

    sl2 = sl * ratio
    img2 = np.zeros((sl2,sl2,3)) 
    return img, lmk

def CropImgPtsGivenResolution(img, lmk, SL, ratio):
    rows,cols,chs = img.shape
    NLMK,dim = lmk.shape
    if dim != 2:
        lmk = lmk.T
        NLMK,dim = lmk.shape
    bbox = Lmk2Bbox(lmk, ratio)
    cx = (bbox[0]+bbox[1]) / 2
    cy = (bbox[2]+bbox[3]) / 2
    sl = bbox[1] - bbox[0]
    scale = 1.* SL[0] / sl
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale)
    img = cv2.warpAffine(img, M, (cols,rows))
    lmk = np.concatenate( (lmk, np.ones((NLMK,1))), axis=1 )
    lmk = np.dot( M, lmk.T )
    lmk = lmk.T # 68x2

    img = img[np.round(cy)-SL[1]/2:np.round(cy)+SL[1]/2, np.round(cx)-SL[0]/2:np.round(cx)+SL[0]/2]
    lmk = lmk - np.tile( np.array([cx-SL[0]/2,cy-SL[1]/2]), (NLMK,1) )
    return (img, lmk)

if __name__=='__main__':
    print 'Python crop image roi by Xi Peng'

    data_path = '/home/ma/xpeng/data/300W/afw/'
    img_list = PXIO.ListFileInFolder(data_path, '.jpg')
    ct = 0
    for img_path in img_list:
        img_name = img_path.split('/')[-1]
        print img_name
        pts_name = img_name[:-4] + '.pts' 
        img = cv2.imread(data_path + img_name)
        lmk = PXPts.Pts2Lmk(data_path + pts_name)
        #lmk = PXPts.Lmk68to7(lmk)

        img2,lmk2 = CropFaceByAllLmk(img, lmk, [100,100], 0.9)
        
        plt.figure(1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        plt.imshow(img2)
        plt.plot(lmk2[:,0], lmk2[:,1], 'og')
        #for i in range(7):
        #    plt.text(lmk2[i,0],lmk2[i,1],str(i),color='red',fontsize=18)
        plt.show()
        ct = ct + 1
        if ct == 20:
            plt.close('all')
            exit()

