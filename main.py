# Xi Peng, Feb 2017
import os, sys, shutil, time, argparse
from PIL import Image, ImageDraw
import numpy as np

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

import path, data, model, util

parser = argparse.ArgumentParser(description='PyTorch res_det Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', dest='pretrained', action='store_false',
                    help='use checkpoint model')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.set_defaults(pretrained=False)

best_rmse = 1
softmax = nn.Softmax2d()
model_save_path = path.model_save_path

def main():
    global args, best_rmse, softmax, model_save_path
    args = parser.parse_args()
    cudnn.benchmark = True

    # create network
    net = model.enc_denc_skip_det_reg_resnet152(args.pretrained)

    # optionally resume from a checkpoint
    if not args.pretrained:
        args.start_epoch,best_rmse = load_checkpoint(net)
    if args.start_epoch == args.epochs:
        args.start_epoch = 0

    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    # Data loading code
    train_list = 'data/train_list.txt'
    train_loader = torch.utils.data.DataLoader(
        data.ImageList( train_list, transforms.ToTensor() ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_list = 'data/val_list.txt'
    val_loader = torch.utils.data.DataLoader(
        data.ImageList( val_list, transforms.ToTensor() ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    weight = torch.Tensor([0.01,1,1,1,1,1,1,1])
    criterion_det = nn.CrossEntropyLoss2d(size_average=True, weight=weight).cuda()
    criterion_reg = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.evaluate:
        validate(val_loader, net, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, net, criterion_det, criterion_reg, optimizer, epoch)

        # evaluate on validation set
        rmse = validate(val_loader, net, criterion_det, criterion_reg)

        # remember best rmse and save checkpoint
        is_best = rmse < best_rmse
        best_rmse = min(rmse, best_rmse)
        save_checkpoint( {'epoch': epoch + 1,
                          'state_dict': net.state_dict(),
                          'best_rmse': best_rmse}, is_best )


def train(train_loader, net, criterion_det, criterion_reg, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_det = AverageMeter()
    losses_reg = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for i, (img, resmap, heatmap, pts7, pts) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_img = torch.autograd.Variable(img)
        resmap = resmap.cuda(async=True)
        heatmap = heatmap.cuda(async=True)
        target_det = torch.autograd.Variable(resmap)
        target_reg = torch.autograd.Variable(heatmap)
        mask_reg = torch.autograd.Variable(heatmap.gt(0)*29).add(1).float()

        # compute output and loss
        out_det,out_reg = net(input_img)
        loss_det = criterion_det(out_det, target_det)
        #loss_reg = criterion_reg(out_reg, target_reg)
        n = heatmap.size(0)*heatmap.size(1)*heatmap.size(2)*heatmap.size(3)
        loss_reg = torch.sum( (out_reg-target_reg)**2 * mask_reg ) / n

        losses_det.update(loss_det.data[0], img.size(0))
        losses_reg.update(loss_reg.data[0], img.size(0))

        loss = loss_det + loss_reg * 0.1
        losses.update(loss.data[0], img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch:[{0}][{1}/{2}] '
                  'Bs:[{3}] '
                  'Time:[{batch_time.val:.3f}]({batch_time.avg:.3f}) '
                  'Data:[{data_time.val:.3f}]({data_time.avg:.3f})\n'
                  'Loss_det:[{loss_det.val:.4f}]({loss_det.avg:.4f}) '
                  'Loss_reg:[{loss_reg.val:.4f}]({loss_reg.avg:.4f}) '
                  'Loss:[{loss.val:.4f}]({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), args.batch_size,
                   batch_time=batch_time, data_time=data_time, 
                   loss_det=losses_det, loss_reg=losses_reg, loss=losses))
            acc = util.per_class_acc_batch(softmax(out_det).cpu().data, resmap.cpu())
            print( 'C0:%.4f C1:%.4f C2:%.4f C3:%.4f C4:%.4f C5:%.4f C6:%.4f C7:%.4f'
                   % (acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7]) )


def validate(val_loader, net, criterion_det, criterion_reg):
    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses_reg = AverageMeter()
    losses = AverageMeter()
    rmses_det = AverageMeter()
    rmses_reg = AverageMeter()

    # switch to evaluate mode
    net.eval()

    end = time.time()
    for i, (img, resmap, heatmap, pts7, pts) in enumerate(val_loader):
        input_img = torch.autograd.Variable(img, volatile=True)
        resmap = resmap.cuda(async=True)
        heatmap = heatmap.cuda(async=True)
        target_det = torch.autograd.Variable(resmap)
        target_reg = torch.autograd.Variable(heatmap)
        mask_reg = torch.autograd.Variable(heatmap.gt(0)*29).add(1).float()

        # compute output and loss
        out_det,out_reg = net(input_img)
        loss_det = criterion_det(out_det, target_det)
        #loss_reg = criterion_reg(out_reg, target_reg)
        n = heatmap.size(0)*heatmap.size(1)*heatmap.size(2)*heatmap.size(3)
        loss_reg = torch.sum( (out_reg-target_reg)**2 * mask_reg ) / n

        losses_det.update(loss_det.data[0], img.size(0))
        losses_reg.update(loss_reg.data[0], img.size(0))

        loss = loss_det + loss_reg * 0.1
        losses.update(loss.data[0], img.size(0))

        # calculate rmse
        pts_det = util.detect_pts(out_det.cpu().data) # b x L x 2
        rmse_det = np.sum(util.per_image_rmse(pts_det, pts7.numpy())) / img.size(0) # b -> 1
        rmses_det.update(rmse_det, img.size(0))

        pts_reg = util.regress_pts(out_reg.cpu().data) # b x L x 2
        rmse_reg = np.sum(util.per_image_rmse(pts_reg, pts.numpy())) / img.size(0) # b -> 1
        rmses_reg.update(rmse_reg, img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % len(val_loader)-1 == 0:
            print('Test:[{0}/{1}] '
                  'Bs:[{2}]'
                  'Time:[{batch_time.val:.3f}]({batch_time.avg:.3f}) '
                  'RMSE_det:[{rmse_det.val:.4f}]({rmse_det.avg:.4f}) '
                  'RMSE_reg:[{rmse_reg.val:.4f}]({rmse_reg.avg:.4f})\n'
                  'Loss_det:[{loss_det.val:.4f}]({loss_det.avg:.4f}) '
                  'Loss_reg:[{loss_reg.val:.4f}]({loss_reg.avg:.4f}) '
                  'Loss:[{loss.val:.4f}]({loss.avg:.4f})'.format(
                   i, len(val_loader), args.batch_size,
                   batch_time=batch_time, rmse_det=rmses_det, rmse_reg=rmses_reg, 
                   loss_det=losses_det, loss_reg=losses_reg, loss=losses))
            acc = util.per_class_acc_batch(softmax(out_det).cpu().data, resmap.cpu())
            print( 'C0:%.4f C1:%.4f C2:%.4f C3:%.4f C4:%.4f C5:%.4f C6:%.4f C7:%.4f'
                   % (acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7]) )
    return rmses_reg.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = model_save_path + filename
    torch.save(state, filename)
    if is_best:
        print("=> saving '{}'".format(model_save_path+'model_best.pth.tar'))
        shutil.copyfile(filename, model_save_path+'model_best.pth.tar')

def load_checkpoint(net, filename='model_best.pth.tar'):
    filename = model_save_path + filename
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        best_rmse = checkpoint['best_rmse']
        #net.load_state_dict(checkpoint['state_dict'])
        state_dict = checkpoint['staite_dict']
        own_state = net.state_dict()
        for name, param in state_dict.items():
            name = name[7:]
            if name not in own_state:
                print('not load weights %s' % name)
                continue
            if isinstance(param, Parameter):
                param = param.data
            own_state[name].copy_(param)
            print('load weights %s' % name)
        print("=> loaded checkpoint '{}'\n=> epoch:{}\tbest_rmse:{}"
              .format(filename, epoch, best_rmse))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return epoch, best_rmse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Epoch:[%d]\tlr:[%f]' % (epoch, lr))

if __name__ == '__main__':
    main()
