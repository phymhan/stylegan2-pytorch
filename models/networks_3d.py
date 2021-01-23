
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

import pdb

dd = pdb.set_trace

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):        
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_D(input_nc, ndf, n_layers_D, n_frames, norm='instance', num_D=1, getIntermFeat=False):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, n_frames, norm_layer, num_D, getIntermFeat)   
    print(netD)
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var

        if input.is_cuda:
            target_tensor = target_tensor.cuda()
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class Relativistic_Average_LSGAN(GANLoss):
    '''
        Relativistic average LSGAN
    '''
    def __call__(self, input_1, input_2, target_is_real):
        if isinstance(input_1[0], list):
            loss = 0
            for input_i, _input_i in zip(input_1, input_2):
                pred = input_i[-1]
                _pred = _input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred - torch.mean(_pred), target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input_1[-1], target_is_real)
            return self.loss(input_1[-1] - torch.mean(input_2[-1]), target_tensor)


class VGGLoss(nn.Module):
    # def __init__(self, vgg_path, gpu_id=0):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()        
        #self.vgg = Vgg19().cuda()
        self.vgg = nn.DataParallel(Vgg19().eval()).cuda()
        #self.vgg = Vgg19().cuda(gpu_id)
        # self.vgg = Vgg19(vgg_path).cuda(gpu_id)
        self.criterion = nn.L1Loss()
        #self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0/32, 1.0/16, 1.0/8]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            #loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss

        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, n_frames=16, norm_layer=nn.BatchNorm3d, 
                    num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
        
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, norm_layer, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)
        if n_frames > 16:
            self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1], count_include_pad=False)
        else:
            self.downsample = nn.AvgPool3d(3, stride=[1, 2, 2], padding=[1, 1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # if use_sigmoid:
        #     sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    # def __init__(self, vgg_path, requires_grad=False):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        
        # vgg19 = models.vgg19(pretrained=False)
        # vgg19.load_state_dict(torch.load(vgg_path))
        # vgg_pretrained_features = vgg19.features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        #self.slice4 = torch.nn.Sequential()
        #self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        #for x in range(12, 21):
        #    self.slice4.add_module(str(x), vgg_pretrained_features[x])
        #for x in range(21, 30):
        #    self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        #h_relu4 = self.slice4(h_relu3)        
        #h_relu5 = self.slice5(h_relu4)                
        #out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        out = [h_relu1, h_relu2, h_relu3]
        return out
