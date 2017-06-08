# Xi Peng, Feb 2017
import os, time
from PIL import Image, ImageDraw
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter

import models.rednet as model
from options.train_options import TrainOptions
from data.load_from_list import ImageLoader
from utils.util import AverageMeter
from utils.util import TrainHistory
from utils.visualizer import Visualizer
from utils.checkpoint import Checkpoint
from pylib import FaceAcc, FacePts, Criterion

def main():
    opt = TrainOptions().parse()
    train_history = TrainHistory()
    checkpoint = Checkpoint(opt)
    visualizer = Visualizer(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    cudnn.benchmark = True

    """build graph"""
    net = model.CreateNet(opt)

    """optimizer"""
    optimizer = model.CreateAdamOptimizer(opt, net)
    #net = torch.nn.DataParallel(net).cuda()
    net.cuda()

    """optionally resume from a checkpoint"""
    checkpoint.load_checkpoint(net, optimizer, train_history)

    """load data"""
    train_list = os.path.join(opt.data_dir, opt.train_list)
    train_loader = torch.utils.data.DataLoader(
        ImageLoader( train_list, transforms.ToTensor(), is_train=True),
        batch_size=opt.bs, shuffle=True,
        num_workers=opt.nThreads, pin_memory=True)

    val_list = os.path.join(opt.data_dir, opt.val_list)
    val_loader = torch.utils.data.DataLoader(
        ImageLoader( val_list, transforms.ToTensor(), is_train=False),
        batch_size=opt.bs, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    """training and validation"""
    for epoch in range(opt.resume_epoch, opt.nEpochs):
        model.AdjustLR(opt, optimizer, epoch)

        # train for one epoch
        train_loss_det, train_loss_reg, tran_loss = \
            train(train_loader, net, optimizer, epoch, visualizer)

        # evaluate on validation set
        val_loss_det, val_loss_reg, val_loss, det_rmse, reg_rmse = \
            validate(val_loader, net, epoch, visualizer, is_show=False)

        # update training history
        e = OrderedDict( [('epoch', epoch)] )
        lr = OrderedDict( [('lr', opt.lr)] )
        loss = OrderedDict( [ ('train_loss_det', train_loss_det),
                              ('train_loss_reg', train_loss_reg),
                              ('val_loss_det', val_loss_det),
                              ('val_loss_reg', val_loss_reg) ] )
        rmse = OrderedDict( [ ('det_rmse', det_rmse), 
                              ('val_rmse', reg_rmse) ] )
        train_history.update(e, lr, loss, rmse)
        checkpoint.save_checkpoint(net, optimizer, train_history)
        visualizer.plot_train_history(train_history)

        # plot best validation
        if train_history.is_best:
            visualizer.imgpts_win_id = 4
            validate(val_loader, net, epoch, visualizer, is_show=True)

def train(train_loader, net, optimizer, epoch, visualizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_det = AverageMeter()
    losses_reg = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for i, (img, pts, gt_det, wt_det, gt_reg) in enumerate(train_loader):
        """forward and backward"""
        #measure data loading time
        data_time.update(time.time() - end)

        # input and groundtruth
        img_var = torch.autograd.Variable(img).cuda(async=True)
        gt_det = gt_det.cuda(async=True)
        gt_det_var = torch.autograd.Variable(gt_det)
        wt_det = wt_det.cuda(async=True)
        wt_det_var = torch.autograd.Variable(wt_det)
        gt_reg = gt_reg.cuda(async=True)
        gt_reg_var = torch.autograd.Variable(gt_reg)

        # detection step
        out_middle,out_det = net(img_var)
        out_det = torch.sigmoid(out_det)
        loss_det = Criterion.weighted_sigmoid_crossentropy( out_det, 
                                                gt_det_var, wt_det_var )
        optimizer.zero_grad()
        loss_det.backward()
        
        # regression step
        out_reg = net(out_det.detach(), out_middle.detach())
        loss_reg = Criterion.L2(out_reg, gt_reg_var)
        loss_reg = loss_reg * 10

        loss_reg.backward()
        optimizer.step()

        """log and display"""
        #measure optimization time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses_det.update(loss_det.data[0])
        losses_reg.update(loss_reg.data[0])
        loss = loss_det + loss_reg
        losses.update(loss.data[0])
        loss_dict = OrderedDict( [ ('loss_det', losses_det.val), 
                        ('loss_reg', losses_reg.val), ('loss', losses.val)] )
        acc = FaceAcc.per_class_f1score(out_det.cpu().data, gt_det.cpu())
        acc_dict = OrderedDict( [('C1',acc[0]), ('C2',acc[1]), ('C3',acc[2]),
                                 ('C4',acc[3]), ('C5',acc[4]), ('C6',acc[5])] )
        visualizer.print_log( 'Train', epoch, i, len(train_loader), 
                              batch_time.avg, value1=loss_dict )

    return losses_det.avg, losses_reg.avg, losses.avg

def validate(val_loader, net, epoch, visualizer, is_show=False):
    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses_reg = AverageMeter()
    losses = AverageMeter()
    rmses_det = AverageMeter()
    rmses_reg = AverageMeter()

    # switch to evaluate mode
    net.eval()

    end = time.time()
    for i, (img, pts, gt_det, wt_det, gt_reg) in enumerate(val_loader):
        """forward only"""
        # input and groundtruth
        img_var = torch.autograd.Variable(img, volatile=True).cuda(async=True)
        gt_det = gt_det.cuda(async=True)
        gt_det_var = torch.autograd.Variable(gt_det, volatile=True)
        wt_det = wt_det.cuda(async=True)
        wt_det_var = torch.autograd.Variable(wt_det, volatile=True)
        gt_reg = gt_reg.cuda(async=True)
        gt_reg_var = torch.autograd.Variable(gt_reg, volatile=True)

        # output and loss 
        out_middle, out_det = net(img_var)
        out_det = torch.sigmoid(out_det)
        loss_det = Criterion.weighted_sigmoid_crossentropy( out_det, 
                                                gt_det_var, wt_det_var )
        
        out_reg = net(out_det, out_middle)
        loss_reg = Criterion.L2(out_reg, gt_reg_var)
        loss_reg = loss_reg * 10

        # calculate rmse
        pred_pts_det = FacePts.Heatmap2Lmk_batch(out_det.cpu().data)
        rmse_det = np.sum(FaceAcc.per_image_rmse( pred_pts_det*4.,
                          FacePts.Lmk68to7_batch(pts.numpy()) )) / img.size(0)  
        pred_pts_reg = FacePts.Heatmap2Lmk_batch(out_reg.cpu().data) # b x L x 2
        rmse_reg = np.sum(FaceAcc.per_image_rmse( pred_pts_reg*2.,
                          pts.numpy() )) / img.size(0)   # b --> 1

        """log and display"""
        #measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses_det.update(loss_det.data[0])
        losses_reg.update(loss_reg.data[0])
        loss = loss_det + loss_reg
        losses.update(loss.data[0])
        rmses_det.update(rmse_det, img.size(0))
        rmses_reg.update(rmse_reg, img.size(0))
        loss_dict = OrderedDict( [('loss_det', losses_det.val), 
                                  ('loss_reg', losses_reg.val),
                                  ('loss', losses.val), 
                                  ('rmse_det', rmses_det.val), 
                                  ('rmse_reg', rmses_reg.val)] )
        acc = FaceAcc.per_class_f1score(out_det.cpu().data, gt_det.cpu())
        acc_dict = OrderedDict( [('C1',acc[0]), ('C2',acc[1]), ('C3',acc[2]),
                                 ('C4',acc[3]), ('C5',acc[4]), ('C6',acc[5])] )
        visualizer.print_log('Val', epoch, i, len(val_loader), batch_time.avg, value1=loss_dict)
        if is_show:
            visualizer.display_imgpts_in_one_batch(img, pred_pts_reg*2.)
    if not is_show:
        return losses_det.avg,losses_reg.avg,losses.avg,rmses_det.avg,rmses_reg.avg,


if __name__ == '__main__':
    main()
