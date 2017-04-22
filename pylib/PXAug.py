# Xi Peng, July 17 2016
import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
#sys.path.append('/media/xpeng/Dropbox/Code/lib/pylib/')
sys.path.append('/Users/Xi/Dropbox/Code/lib/pylib/')
import PXPath, PXIO, PXPts

def FlipImgLmk_7lmk(img,lmk):
    img2 = np.copy(img)
    lmk2 = np.copy(lmk)
    NLMK = 7
    rows,cols,chs = img.shape
    img2 = cv2.flip(img,1)

    lmk2[:,0] = np.tile(cols-1,(NLMK,1)).T - lmk2[:,0]
    lmk3 = np.copy(lmk2)
    lmk2[0,:] = lmk3[3,:]
    lmk2[1,:] = lmk3[2,:]
    lmk2[2,:] = lmk3[1,:]
    lmk2[3,:] = lmk3[0,:]
    lmk2[4,:] = lmk3[4,:]
    lmk2[5,:] = lmk3[6,:]
    lmk2[6,:] = lmk3[5,:]
    return (img2,lmk2)

def SampleFaceAugment(img, lmk, SL, EM_RATIO, isaug):
    if isaug[0]:
        isflip = np.random.choice(2)
        if isflip:
            img2,lmk2 = FlipImgLmk_7lmk(img,lmk)
        else:
            img2 = np.copy(img)
            lmk2 = np.copy(lmk)
    else: 
        isflip = 0
        img2 = np.copy(img)
        lmk2 = np.copy(lmk)
    if isaug[1]:
        scale = np.random.normal(1.0, 0.3) # 0.7 - 1.3
    else:
        scale = 1.0
    if isaug[2]:
        trans = np.round( np.random.normal(0, SL[0]*0.05, 2) ) # -8.96 - 8.96
    else:
        trans = [0,0]
    #print 'flip:%d scale:%.3f trans:%.1f %.1f' % (isflip,scale,trans[0],trans[1])
    
    NLMK = lmk.shape[0]
    if NLMK == 7:
        cx1,cy1,dist1 = PXPts.GetCenterDist_7lmk(lmk2)
    elif NLMK == 68:
        cx1,cy1,dist1 = PXPts.GetCenterDist_68lmk(lmk2)

    if len(img2.shape) == 2:
        height,width = img2.shape
        img2 = np.resize(img2, (height,width,3))
    rows,cols,chs = img2.shape
    scale = scale * ( SL[1] / (dist1*EM_RATIO) )
    M = cv2.getRotationMatrix2D((cx1,cy1), 0, scale)
    img2 = cv2.warpAffine(img2, M, (cols,rows))
    # enlarge
    img_en = np.uint8(np.zeros((rows+2*SL[0],cols+2*SL[1],3)))
    img_en[SL[0]:rows+SL[0], SL[1]:cols+SL[1], :] = img2
    
    lmk2 = np.concatenate( (lmk2,np.ones((NLMK,1))), axis=1 ) 
    lmk2 = np.dot( M, lmk2.T )
    lmk2 = lmk2.T # L x 2
    # enlarge
    lmk_en = lmk2 + np.tile( SL,(NLMK,1) )
    if NLMK == 7: 
        cx2,cy2,dist2 = PXPts.GetCenterDist_7lmk(lmk_en)
    elif NLMK ==68:
        cx2,cy2,dist2 = PXPts.GetCenterDist_68lmk(lmk_en)
        
    cx3 = cx2 + trans[0]
    cy3 = cy2 + trans[1]

    img3 = img_en[np.round(cy3)-SL[1]/2:np.round(cy3)+SL[1]/2, np.round(cx3)-SL[0]/2:np.round(cx3)+SL[0]/2]
    lmk3 = lmk_en - np.tile( np.array([cx3-SL[0]/2,cy3-SL[1]/2]), (NLMK,1) )

    return (img3, lmk3)
    
    
if __name__=='__main__':
    print 'Python Data Augmentation by Xi Peng'

    data_path = PXPath.CurDataPath() + '/resource/face_align/300W/lfpw/'
    print data_path
    img_list = PXIO.ListFileInFolder(data_path, '.jpg')
    for img_path in img_list:
        img_name = img_path.split('/')[-1]
        #img_name = 'lfpw_test_image_0013.jpg'
        pts_name = img_name[:-4] + '.pts' 
        img = cv2.imread(data_path + img_name)
        lmk = PXPts.Pts2Lmk(data_path + pts_name)
        lmk = PXPts.Lmk68to7(lmk)

        for i in range(10):
            img2,lmk2 = SampleAugment_7lmks(img, lmk, [128,128],0.3,[1,1,1])
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            if 0:
                plt.figure(1)
                plt.subplot(121)
                plt.imshow(img)
                plt.plot(lmk[:,0], lmk[:,1], 'og')
                plt.subplot(122)
                plt.imshow(img2)
                plt.plot(lmk2[:,0], lmk2[:,1], 'og')
                for i in range(7):
                    plt.text(lmk2[i,0],lmk2[i,1],str(i),color='red',fontsize=18)
                plt.show()
                plt.close('all')

