# Xi Peng, Aug 27 2016
import os, sys
import numpy as np
import h5py

def CreateMatrixFloat32(bs, ch, w, h):
    #total_size = bs * ch * h * w
    #data = np.(total_size)
    #data = data.reshape(bs, ch, h, w)
    #data = data.astype('float32') 
    data = np.zeros((bs,ch,h,w), dtype='float32')
    return data

def CreateMatrixUint8(bs, ch, w, h):
    #total_size = bs * ch * h * w
    #data = np.arange(total_size)
    #data = data.reshape(bs, ch, h, w)
    #data = data.astype('uint8') 
    data = np.zeros((bs,ch,h,w), dtype='uint8')
    return data

def CreateVectorFloat32(bs, num):
    #label = np.arange(num * bs)
    #label = label.reshape(bs, num)
    #label = label.astype('float32')
    data = np.zeros((bs,num), dtype='float32')
    return data

def CreateVectorUint8(bs, num):
    #label = np.arange(num * bs)
    #label = label.reshape(bs, num)
    #label = label.astype('uint8')
    data = np.zeros((bs,num), dtype='uint8')
    return data

def SetImgMatrix255(data, index, img, wid, hei):
    if len(img.getbands()) == 1:
        I = img.getdata()
        I = np.array(I, dtype='float32').reshape((wid,hei))
        c_img = np.asarray([I])
        data[index, :, :, :] = c_img
    elif len(img.getbands()) == 3:
        R,G,B = img.getdata(0), img.getdata(1), img.getdata(2)
        R = np.array(R, dtype='float32').reshape((wid,hei))
        G = np.array(G, dtype='float32').reshape((wid,hei))
        B = np.array(B, dtype='float32').reshape((wid,hei))
        #c_img = np.asarray([R,G,B]) / 255.
        c_img = np.asarray([R,G,B])
        data[index, :, :, :] = c_img
    return

def SetVector(label, index, tl):
    label[index, :] = tl
    return

if __name__ == "__main__":
    print 'Genarate HDF5 file'
