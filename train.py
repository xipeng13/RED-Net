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

from options.train_options import TrainOptions
from data.load_from_list import ImageLoader
from models.rednet_reg import CreateNet
from utils.util import AverageMeter
from utils.util import TrainHistory
from utils.visualizer import Visualizer
from utils.checkpoint import Checkpoint
from pylib import FaceAcc, FacePts, Criterion
cudnn.benchmark = True

def main():
    opt = TrainOptions().parse()
    train_history = TrainHistory()
    checkpoint = Checkpoint(opt)
    visualizer = Visualizer(opt)

    """optionally resume from a checkpoint"""
    net = CreateNet(opt)
    checkpoint.load_checkpoint(net, train_history)

    """load data"""
    net = torch.nn.DataParallel(net).cuda()
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

    """optimizer"""
    #optimizer = torch.optim.SGD( net.parameters(), lr=opt.lr,
    #                             momentum=opt.momentum,
    #                             weight_decay=opt.weight_decay )
    optimizer = torch.optim.Adam( net.parameters(), lr=opt.lr,
                                  betas=(opt.beta1,0.999))

    """training and validation"""
    for epoch in range(opt.resume_epoch, opt.nEpochs):
        # train for one epoch
        train_loss = train(train_loader, net, optimizer, epoch, visualizer)

        # evaluate on validation set
        val_loss, val_rmse = validate(val_loader, net, epoch, visualizer)

        # update training history
        e = OrderedDict( [('epoch', epoch)] )
        lr = OrderedDict( [('lr', opt.lr)] )
        loss = OrderedDict( [('train_loss', train_loss),('val_loss', val_loss)] )
        rmse = OrderedDict( [('val_rmse', val_rmse)] )
        train_history.update(e, lr, loss, rmse)
        checkpoint.save_checkpoint(net, train_history)
        visualizer.plot_train_history(train_history)


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
        """measure data loading time"""
        data_time.update(time.time() - end)

        # input and groundtruth
        input_img = torch.autograd.Variable(img)

        gt_det = gt_det.cuda(async=True)
        gt_det_var = torch.autograd.Variable(gt_det)
        wt_det = wt_det.cuda(async=True)
        wt_det_var = torch.autograd.Variable(wt_det)
        gt_reg = gt_reg.cuda(async=True)
        gt_reg_var = torch.autograd.Variable(gt_reg)

        # output and loss
        output = net(input_img)
        #pred_det = torch.sigmoid(output)
        #loss_det = Criterion.weighted_sigmoid_crossentropy(pred_det, gt_det_var, wt_det_var)
        pred_reg = output
        loss_reg = Criterion.L2(pred_reg, gt_reg_var)
        loss = loss_reg

        # gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """measure optimization time"""
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses_reg.update(loss_reg.data[0])
        losses.update(loss.data[0])
        loss_dict = OrderedDict( [('loss_reg', losses_reg.val), ('loss', losses.val)] )
        #acc = FaceAcc.per_class_f1score(pred_det.cpu().data, gt_det.cpu())
        #acc_dict = OrderedDict( [('C1',acc[0]), ('C2',acc[1]), ('C3',acc[2]),
        #                         ('C4',acc[3]), ('C5',acc[4]), ('C6',acc[5])] )
        visualizer.print_log( 'Train', epoch, i, len(train_loader), batch_time.avg,
                              value1=loss_dict )

    return losses.avg

def validate(val_loader, net, epoch, visualizer):
    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses_reg = AverageMeter()
    losses = AverageMeter()
    rmses_reg = AverageMeter()

    # switch to evaluate mode
    net.eval()

    end = time.time()
    for i, (img, pts, gt_det, wt_det, gt_reg) in enumerate(val_loader):
        # input and groundtruth
        input_img = torch.autograd.Variable(img, volatile=True) # b x 3 x W x H

        gt_det = gt_det.cuda(async=True)
        gt_det_var = torch.autograd.Variable(gt_det)
        wt_det = wt_det.cuda(async=True)
        wt_det_var = torch.autograd.Variable(wt_det)
        gt_reg = gt_reg.cuda(async=True)
        gt_reg_var = torch.autograd.Variable(gt_reg)

        # output and loss
        output = net(input_img)
        #pred_det = torch.sigmoid(output)
        #loss_det = Criterion.weighted_sigmoid_crossentropy(pred_det, gt_det_var, wt_det_var)
        pred_reg = output
        loss_reg = Criterion.weighted_L2(pred_reg, gt_reg_var)
        loss = loss_reg

        # calculate rmse
        pred_pts_reg = FacePts.Heatmap2Lmk_batch(pred_reg.cpu().data) # b x L x 2
        rmse_reg = np.sum(FaceAcc.per_image_rmse( pred_pts_det*2.,
                          pts.numpy() )) / img.size(0)   # b --> 1

        """measure elapsed time"""
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses_reg.update(loss_reg.data[0])
        losses.update(loss.data[0])
        rmses_reg.update(rmse_reg, img.size(0))
        loss_dict = OrderedDict( [('loss_det', losses_reg.val), ('loss', losses.val),
                                  ('rmse_det', rmses_reg.val)] )
        #acc = FaceAcc.per_class_f1score(pred_det.cpu().data, gt_det.cpu())
        #acc_dict = OrderedDict( [('C1',acc[0]), ('C2',acc[1]), ('C3',acc[2]),
        #                         ('C4',acc[3]), ('C5',acc[4]), ('C6',acc[5])] )
        visualizer.print_log( 'Val', epoch, i, len(val_loader), batch_time.avg,
                              value1=loss_dict )
        if i==0:
            visualizer.display_imgpts_in_one_batch(img, pred_pts_reg*2.)

    return losses.avg, rmses_reg.avg,



if __name__ == '__main__':
    main()
