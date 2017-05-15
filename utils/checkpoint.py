# Xi Peng, May 2017
import os, shutil
import torch

class Checkpoint():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)

    def save_checkpoint(self, net, train_history):
        save_name = 'ep%02d.pth.tar' % train_history.epoch[-1]['epoch']
        save_path = os.path.join(self.save_dir, save_name)

        checkpoint = { 'train_history': train_history.state_dict(), 
                       'state_dict': net.state_dict()}
        torch.save( checkpoint, save_path )
        print("=> saving '{}'".format(save_path))
        if train_history.is_best:
            print("=> saving '{}'".format(self.save_dir+'/model_best.pth.tar'))
            save_path2 = os.path.join(self.save_dir, 'model_best.pth.tar')
            shutil.copyfile(save_path, save_path2)

    def load_checkpoint(self, net, train_history):
        if not self.opt.load_checkpoint:
            return
        
        save_name = 'ep%02d.pth.tar' % self.opt.resume_epoch
        save_path = os.path.join(self.save_dir + save_name)
        if os.path.isfile(save_path):
            print("=> loading checkpoint '{}'".format(save_path))
            checkpoint = torch.load(save_path)

            train_history.load_state_dict( checkpoint['train_history'] )
            #net.load_state_dict(checkpoint['state_dict'])
            state_dict = checkpoint['staite_dict']

            net_dict = net.state_dict()
            for name, param in state_dict.items():
                name = name[7:] #????????????????
                if name not in net_dict:
                    print("=> not load weights '{}'".format(name))
                    continue
                if isinstance(param, Parameter):
                    param = param.data
                net_dict[name].copy_(param)
                print("load weights '{}'".format(name))
            print( "=> loaded checkpoint '{}'\t=> epoch:{}"
                  .format(save_name, self.opt.resume_epoch) )
        else:
            print("=> no checkpoint found at '{}'".format(save_path))
  
