# Xi Peng, Feb 2017
import os, sys
import numpy as np
from PIL import Image
import torch
import PXPts


def GrayPILImageToTensor255(img):
    # [0, L), h x w
    tensor = np.asarray(img).reshape( (img.size[1], img.size[0]) )
    tensor = torch.from_numpy(tensor)
    return tensor

def Tensor255ToGrayPILImage(tensor, scale=30):
    img = tensor.mul(scale).byte().numpy()
    img = Image.fromarray(img)
    return img

def Tensor01ToGrayPILImage(tensor):
    img = tensor.mul(255).byte().numpy()
    img = Image.fromarray(img)
    return img

def per_class_acc_batch(output, target):
    # output: softmax output b x c x h x w tensor [0,1]
    # target: class annotation b x h x w tensor [0,C-1)
    batch_size = target.size(0)
    num_class = output.size(1)
    output, target = output.numpy(), target.numpy()
    output = np.argmax(output, axis=1)

    acc_arr_sum = np.zeros(num_class)
    for b in range(batch_size):
        pred = np.squeeze(output[b,])
        ann = np.squeeze(target[b,])
        acc_arr = per_class_acc_single(pred, ann, num_class)
        acc_arr_sum += acc_arr
    return acc_arr_sum/batch_size

def per_class_acc_single(pred, ann, num_class):
    # pred: argmax output h x w numpy [0,C-1)
    # ann: class annotation h x w numpy [0,C-1)
    acc_arr = np.zeros(num_class)
    for c in range(num_class):
        idx = np.where(ann==c)
        idx_match = np.where(pred[idx]==c)
        acc = 1.0 * len(idx_match[0]) / len(idx[0])
        acc_arr[c] = acc
    return acc_arr

def detect_pts(output):
    # output: softmax output b x c x h x w tensor [0,1]
    # pts: b x num_pts x 2 numpy
    batch_size, num_class = output.size(0), output.size(1)
    num_pts = num_class - 1

    output = np.argmax(output.numpy(), axis=1) # b x h x w
    pts = np.zeros((batch_size, num_pts, 2))
    for b in range(batch_size):
        resmap = np.squeeze(output[b,]) # h x w
        pts[b,] = PXPts.Resmap2Lmk(resmap, num_pts)
    return pts   

def regress_pts(output):
    # output: output b x num_pts x h x w tensor [0,1]
    # pts: b x num_pts x 2 numpy
    batch_size, num_pts = output.size(0), output.size(1)
    
    pts = np.zeros((batch_size, num_pts, 2))
    for b in range(batch_size):
        heatmap = output[b,].numpy() # c x h x w
        pts[b,] = PXPts.Heatmap2Lmk(heatmap)
    return pts

def regress_pts2(output):
    # output: output b x c x h x w tensor [0,1]
    # pts: b x num_pts x 2 numpy
    b,c,h,w = output.size()
    max_score,idx = torch.max(output.view(b,c,h*w), 2)
    pts = idx.repeat(1,1,2).float()
    pts[:,:,0] = pts[:,:,0] % w
    pts[:,:,1] = (pts[:,:,1] / w).floor()
    mask = max_score.gt(0).repeat(1,1,2).float()
    #pts = pts.mul(mask).add(1)
    pts = pts.add(1)
    return pts.numpy()

def per_image_rmse(pred, ann):
    # pred: N x L x 2 numpy
    # ann:  N x L x 2 numpy
    # rmse: N numpy 
    N = pred.shape[0]
    L = pred.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = pred[i,], ann[i,]
        if L == 7:
            interocular = np.linalg.norm(pts_gt[0,]-pts_gt[3,])
        elif L == 68:
            interocular = np.linalg.norm(pts_gt[36,]-pts_gt[45,])
        rmse[i] = np.sum(np.linalg.norm(pts_pred-pts_gt, axis=1))/(interocular*L)
    return rmse
 
def per_image_rmse_component(pred, ann):
    # pred: N x L x 2 numpy
    # ann:  N x L x 2 numpy
    # rmse: N numpy 
    N = pred.shape[0]
    L = pred.shape[1]
    rmse = np.zeros(N)
    rmse_le = np.zeros(N)
    rmse_re = np.zeros(N)
    rmse_ns = np.zeros(N)
    rmse_mt = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = pred[i,], ann[i,]
        if L == 7:
            interocular = np.linalg.norm(pts_gt[0,]-pts_gt[3,])
        elif L == 68:
            interocular = np.linalg.norm(pts_gt[36,]-pts_gt[45,])
        rmse[i] = np.sum(np.linalg.norm(pts_pred-pts_gt, axis=1))/(interocular*L)
        if L == 7:
            rmse_le[i] = np.sum(np.linalg.norm(pts_pred[0:2,]-pts_gt[0:2,], axis=1))/(interocular*2)
            rmse_re[i] = np.sum(np.linalg.norm(pts_pred[2:4,]-pts_gt[2:4,], axis=1))/(interocular*2)
            rmse_ns[i] = np.sum(np.linalg.norm(pts_pred[4,]-pts_gt[4,], axis=1))/(interocular*1)
            rmse_mt[i] = np.sum(np.linalg.norm(pts_pred[5:7,]-pts_gt[5:7,], axis=1))/(interocular*2)
        elif L == 68:
            rmse_le[i] = np.sum(np.linalg.norm(pts_pred[36:42,]-pts_gt[36:42,], axis=1))/(interocular*6)
            rmse_re[i] = np.sum(np.linalg.norm(pts_pred[42:48,]-pts_gt[42:48,], axis=1))/(interocular*6)
            rmse_ns[i] = np.sum(np.linalg.norm(pts_pred[27:36,]-pts_gt[27:36,], axis=1))/(interocular*9)
            rmse_mt[i] = np.sum(np.linalg.norm(pts_pred[48:68,]-pts_gt[48:68,], axis=1))/(interocular*20)
    return rmse,rmse_le,rmse_re,rmse_ns,rmse_mt
 

def save_image_resmap_heatmap(img, resmap, heatmap, num_pts):
    # Debug
    ToPILImage = transforms.ToPILImage()
    for b in range(args.batch_size):
        img = ToPILImage(img[b,])
        img.save('img_%d.jpg' % b)
        resmap = util.Tensor255ToGrayPILImage(resmap[b,], scale=30)
        resmap.save('res_%d.png' % b)
        for c in range(num_pts):
            #heatmap = util.Tensor01ToGrayPILImage(heatmap[b,c,])
            heatmap = util.Tensor255ToGrayPILImage(heatmap[b,c,], scale=2)
            heatmap.save("heatmap_%d_%d.png" % (b,c))


