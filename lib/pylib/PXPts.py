import os
import numpy as np
from PIL import Image, ImageDraw

## read and write
def ReadLmkFromTxt(path,format):
    ct = 0
    list = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if file.endswith(format):
                lmk = np.loadtxt(root+file) # 68x2
                n,d = lmk.shape
                lmk = lmk.reshape((n*d))
                list.append(lmk)
                ct = ct + 1
                if ct == 10000:
                    return list
    return list

def ReadLmkFromTxtRecursive(path,format):
    ct = 0
    list = []
    for root, dirs, files in os.walk(path):
        for fold in dirs:
            files = os.listdir(root+fold)
            for file in sorted(files):
                if file.endswith(format):
                    lmk = np.loadtxt(root+fold+'/'+file) # 68x2
                    n,d = lmk.shape
                    lmk = lmk.reshape((n*d))
                    list.append(lmk)
                    ct = ct + 1
                    if ct == 10000:
                        return list
    return list

## pts v.s. lmk
def Pts2Lmk(fname):
    n_lmk = 68
    lmk = np.genfromtxt(fname, delimiter=' ', skip_header=3, skip_footer=1)
    return lmk

def Lmk68to7(lmk):
    lmk2 = np.zeros( (7,2) )
    lmk2[0] = lmk[37-1]
    lmk2[1] = lmk[40-1]
    lmk2[2] = lmk[43-1]
    lmk2[3] = lmk[46-1]
    lmk2[4] = lmk[31-1]
    lmk2[5] = lmk[49-1]
    lmk2[6] = lmk[55-1]
    return lmk2

def GetCenterDist_68lmk(lmk):
    eyec = np.mean(lmk[36:48,:], axis=0)
    mouc = np.mean(lmk[48:60,:], axis=0)
    eyec_mouc_dist = np.sqrt(np.sum((eyec-mouc)**2))
    cx = int((eyec[0]+mouc[0]) / 2)
    cy = int((eyec[1]+mouc[1]) / 2)
    return (cx, cy, eyec_mouc_dist)

def GetCenterDist_7lmk(lmk):
    eyec = np.mean(lmk[0:4,:], axis=0)
    mouc = np.mean(lmk[5:7,:], axis=0)
    eyec_mouc_dist = np.sqrt(np.sum((eyec-mouc)**2))
    cx = int((eyec[0]+mouc[0]) / 2)
    cy = int((eyec[1]+mouc[1]) / 2)
    return (cx, cy, eyec_mouc_dist)

def Lmk2Bbox_7lmk(lmk, DISTRATIO):
    cx,cy,dist = GetCenterDist_7lmk(lmk)
    sl = int(dist * DISTRATIO)
    bbox = (cx-sl/2, cy-sl/2, cx+sl/2, cy+sl/2) # left, top, right, bottom 
    return bbox

## resmap
def Lmk2Resmap(lmk, shape, circle_size):
    #RADIUS = GetCircleSize_L128_R4(scale)
    RADIUS = circle_size
    resmap = Image.new('L', shape)
    draw = ImageDraw.Draw(resmap)
    for l in range(lmk.shape[0]):
        draw.ellipse((lmk[l,0]-RADIUS,lmk[l,1]-RADIUS,lmk[l,0]+RADIUS,lmk[l,1]+RADIUS), fill=l+1)
    del draw
    return resmap

def Resmap2Lmk(resmap, NLMK):
    # resmap: h x w numpy [0,NLMK)
    lmk = np.zeros((NLMK, 2))
    for l in range(NLMK):
        try:
            y,x = np.where(resmap == l+1)
            yc,xc = np.mean(y), np.mean(x)
            lmk[l,:] = [xc+1, yc+1]
        except:
            print('Not found %d-th landmark' % l)
    return lmk

def GetCircleSize_L128_R4(scale):
    size = np.round( 4 / scale )
    if size<2:
        size = 2
    elif size>5:
        size = 5
    return size

def CircleSize(base_size=4, scale=1):
    size = np.round( base_size / scale) # L128-R4
    size = size-2 if size<base_size-2 else size
    size = size+2 if size>base_size+2 else size
    return size

## heatmap
def draw_gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    tmp_size = np.round(3 * sigma)
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size), int(pt[1] + tmp_size)]
    # Check that any part of the gaussian is in-bounds
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def Lmk2Heatmap(pts, res, sigma=1):
    # generate heatmap n x res[1] x res[0], each row is one pt (x, y)
    heatmap = np.zeros((pts.shape[0], res[0], res[1]))
    for i in range(0, pts.shape[0]):
        if pts[i,0]>0 and pts[i,0]<=res[0] and pts[i,1]>0 and pts[i,1]<=res[1]:
            heatmap[i] = draw_gaussian(heatmap[i], pts[i,], sigma)
    return heatmap

def Heatmap2Lmk(heatmap):
    # heatmap: NLMK x h x w numpy [0,1]
    # lmk: NLMK x 2 numpy
    NLMK = heatmap.shape[0]
    lmk = np.zeros((NLMK, 2))
    for l in range(NLMK):
        y,x = np.unravel_index(heatmap[l,].argmax(), heatmap[l,].shape)
        lmk[l,:] = [x+1, y+1]
    return lmk


if __name__=='__main__':
    print 'Python pts to landmark by Xi Peng'

