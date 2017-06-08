# Xi Peng, Feb 2017
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from pylib.RecurrentBN import RBN2d


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class Conv1x1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Residual(nn.Module):
    expansion = 4
    def __init__(self, ch_in, ch, stride=1, downsample=None, is_recur=False):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.conv1 = nn.Conv2d(ch_in, ch, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(ch, ch * 4, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(ch_in, ch * 4, kernel_size=1, bias=False)

        if is_recur:
            self.bn1 = RBN2d(ch)
            self.bn2 = RBN2d(ch)
            self.bn3 = RBN2d(ch * 4)
            self.bn4 = RBN2d(ch * 4)
        else:
            self.bn1 = nn.BatchNorm2d(ch)
            self.bn2 = nn.BatchNorm2d(ch)
            self.bn3 = nn.BatchNorm2d(ch * 4)
            self.bn4 = nn.BatchNorm2d(ch * 4)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(1) != out.size(1):
            residual = self.conv4(residual)
            residual = self.bn4(residual)

        out += residual
        out = self.relu(out)
        return out

class Deconv(nn.Module):
    expansion = 2
    def __init__(self, ch_in, ch, is_output_relu=True):
        super(Deconv, self).__init__()
        self.dconv = nn.ConvTranspose2d(ch_in, ch, kernel_size=2, 
                                        stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv = nn.Conv2d(ch, ch * 2, 
                              kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch * 2)
        self.relu = nn.ReLU(inplace=True)
        self.is_output_relu = is_output_relu

    def forward(self, x):
        out = self.dconv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn2(out)
        if self.is_output_relu == True:
            out = self.relu(out)
        return out

class Upsample(nn.Module):
    expansion = 2
    def __init__(self, ch_in, ch_out, is_recur=False):
        super(Upsample, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(ch_in, ch_out, 
                              kernel_size=1, bias=False)
        if is_recur:
            self.bn = RBN2d(ch_out)
        else:
            self.bn = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv(out)
        out = self.bn(out)      # no relu here
        return out
        
        
class recurrent_det_reg(nn.Module):
    def __init__(self, block, layers):
        self.ch_in = 64
        super(recurrent_det_reg, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)      # 64x64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, is_recur=True)  # 32x32
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, is_recur=True)  # 16x16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_recur=True)  # 8x8

        # upsample
        self.skip1 = self._stack_residual(block, 256, 64, 3, stride=1, is_recur=True)   # 64x64,256
        self.skip2 = self._stack_residual(block, 512, 128, 3, stride=1, is_recur=True)	# 32x32,128
        self.skip3 = self._stack_residual(block, 1024, 256, 3, stride=1, is_recur=True) # 16x16,1024

        self.upsample4 = Upsample(2048, 1024, is_recur=True)# 16x16,1024 
        self.upsample3 = Upsample(1024, 512, is_recur=True) # 32x32,512
        self.upsample2 = Upsample(512, 256, is_recur=True)	# 64x64,256
        self.upsample1 = Upsample(256, 128) 	# 128x128,128

        self.dlayer3 = self._stack_residual(block, 1024, 256, 3,stride=1,is_recur=True) # 16x16,1024
        self.dlayer2 = self._stack_residual(block, 512, 128, 3,stride=1,is_recur=True)  # 32x32,512
        self.dlayer1 = self._stack_residual(block, 256, 64, 3, stride=1,is_recur=True)  # 64x64,256
        self.dlayer0 = self._stack_residual(block, 128, 32, 3, stride=1)    # 128x128,128

        self.fc_det = Conv1x1(256, 64)    # 64x64
        self.out_det = nn.Conv2d(64, 7, kernel_size=1, stride=1, bias=False)
        self.det_fb = self._stack_residual(block, 7, 64, 1, stride=1)    # 64x64,256

        self.out_reg = nn.Conv2d(128, 68, kernel_size=1, stride=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
				#m.weight.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, ch, num_block, stride=1, is_recur=False):
        downsample = None
        if stride != 1 or self.ch_in != ch * block.expansion:
            if is_recur:
                downsample = nn.Sequential(
                    nn.Conv2d(self.ch_in, ch * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    RBN2d(ch * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.ch_in, ch * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(ch * block.expansion),
                )

        layers = []
        layers.append(block(self.ch_in, ch, stride, downsample, is_recur=is_recur))
        self.ch_in = ch * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.ch_in, ch, is_recur=is_recur))

        return nn.Sequential(*layers)

    def _stack_residual(self, block, ch_in, ch, num_block, stride=1, is_recur=False):
        layers = []
        layers.append(block(ch_in, ch, stride=1, downsample=None, is_recur=is_recur))
        for i in range(1, num_block):
            layers.append(block(ch * 4, ch, is_recur=is_recur))

        return nn.Sequential(*layers)

    def forward(self, x, x_middle=None):
        if x.size(1) == 3:
            if_detection = True
        elif x.size(1) == 7:
            if_detection = False

        if if_detection:
            # conv1 & layer1
            x = self.conv1(x)   # 128 x 128
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x) # 64 x 64
            x = self.layer1(x)  # 64 x 64
            x_middle = x
        else:
            x = self.det_fb(x)
            x = x + x_middle

        s1 = self.skip1(x)  # 64 x 64
        x = self.layer2(x)  # 32 x 32
        s2 = self.skip2(x)  # 32 x 32
        x = self.layer3(x)  # 16 x 16
        s3 = self.skip3(x)  # 16 x 16
        x = self.layer4(x)  # 8 x 8

        # decoder
        x = self.upsample4(x) # 16 x 16
        x += s3
        x = self.relu(x)
        x = self.dlayer3(x)

        x = self.upsample3(x) # 32 x 32
        x += s2
        x = self.relu(x)
        x = self.dlayer2(x)

        x = self.upsample2(x) # 64 x 64
        x += s1
        x = self.relu(x)
        x = self.dlayer1(x)

        if if_detection:
            # detection
            x_fc_det = self.fc_det(x)
            x_det = self.out_det(x_fc_det)
            return x_middle, x_det
        else:
            # regression
            x = self.upsample1(x) # 128 x 128
            x = self.relu(x)
            x = self.dlayer0(x)
            x_reg = self.out_reg(x)
            return x_reg


def CreateNet(opt):
    net = recurrent_det_reg(Residual, [3, 4, 6, 3]) # ResNet-50
    net_dict = net.state_dict()
    #for name, param in net_dict.items():
    #    print(name)
    #    print(param.size())

    if opt.load_pretrained:
        print('=>load pretrained weights ...')
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        for name, param in state_dict.items():
            if name not in net_dict:
                print('not load weights %s' % name)
                continue
            if isinstance(param, Parameter):
                param = param.data
            net_dict[name].copy_(param)
            #print('load weights %s' % name)
    return net

def CreateAdamOptimizer(opt, net):
    optimizer = torch.optim.Adam( [
        {'params': net.conv1.parameters(),  'lr': 0},
        {'params': net.bn1.parameters(),    'lr': 0},
        {'params': net.layer1.parameters(), 'lr': 0},
        {'params': net.layer2.parameters(), 'lr': opt.lr*0.1},
        {'params': net.layer3.parameters(), 'lr': opt.lr*0.1},
        {'params': net.layer4.parameters(), 'lr': opt.lr*0.1},
        {'params': net.skip1.parameters()},
        {'params': net.skip2.parameters()},
        {'params': net.skip3.parameters()},
        {'params': net.dlayer3.parameters()},
        {'params': net.dlayer2.parameters()},
        {'params': net.dlayer1.parameters()},
        {'params': net.dlayer0.parameters()},
        {'params': net.upsample4.parameters()},
        {'params': net.upsample3.parameters()},
        {'params': net.upsample2.parameters()},
        {'params': net.upsample1.parameters()},
        {'params': net.fc_det.parameters()},
        {'params': net.out_det.parameters()},
        {'params': net.det_fb.parameters()},
        {'params': net.out_reg.parameters()},
    ], lr=opt.lr, betas=(opt.beta1,0.999) )
    return optimizer

def AdjustLR(opt, optimizer, epoch):
    if epoch < 30:
    	for param_group in optimizer.param_groups:
        	print(param_group['lr'])
        return
    elif epoch == 30:
         opt.lr = opt.lr * 0.1
    elif epoch == 60:
         opt.lr = opt.lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr
        print(param_group['lr'])

