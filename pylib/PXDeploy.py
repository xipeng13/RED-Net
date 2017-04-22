# Xi Peng, July 20 2016
import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/net/ca-home/vol/mai/xpeng/lib/pylib/')
import PXIO, PXPts
CAFFE_ROOT =  '/home/ma/xpeng/caffe_local/'
sys.path.insert(0, CAFFE_ROOT+'python')
import caffe

def DeployBatch(net, layer_in, layer_out, data_in, bs_in, dim_out):
    num, ch, wid, hei = data_in.shape
    bs = bs_in
    fea_dim = dim_out
    data_out = np.arange(num * fea_dim)
    data_out = data_out.reshape(num, fea_dim)
    data_out = data_out.astype('float32')
    data = np.arange(bs * ch * hei * wid)
    data = data.reshape(bs, ch, hei, wid)
    data = data.astype('float32')

    for i in range(num/bs + 1):
        num_cur = num - i*bs
        if num_cur >= bs:
            data = data_in[i*bs:(i+1)*bs,:,:,:]
        else:
            data[0:num_cur,:,:,:] = data_in[i*bs:num,:,:,:]
            for r in range(bs-num_cur):
                data[num_cur+r,:,:,:] = data_in[num-1,:,:,:]

        net.blobs[layer_in].data[...] = data
        net.forward()
        res = net.blobs[layer_out].data[...]

        if num_cur >= bs:
            data_out[i*bs:(i+1)*bs,:] = res
        else:
            data_out[i*bs:num,:] = res[0:num_cur,:]
    return data_out

if __name__=='__main__':
    print 'Python caffe deploy by Xi Peng'

    workpath = '/net/ca-home/vol/mai/xpeng/casia/pose_estimate/'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(workpath+'deploy.prototxt', workpath+'25_06_2016_iter_500000.caffemodel', caffe.TEST)
    
    img_list = PXIO.ListFileInFolder('/net/acadia1a/data/xpeng/data/casia/img_cp0.9/0000045/', '.jpg')
    num = len(img_list)
    ch, hei, wid = 3, 100, 100
    data_in = np.arange(num * ch * hei * wid)
    data_in = data_in.reshape(num, ch, hei, wid)
    data_in = data_in.astype('float32')

    for i in range(num):
        im_path = img_list[i]
        im_name = im_path.split('/')[-1]
        print im_name
        im = caffe.io.load_image(im_path)
        data_in[i,:,:,:] = np.transpose(im, (2,0,1))
    
    data_out = DeployBatch(net,'data','fc7_pos',data_in,100,7)

