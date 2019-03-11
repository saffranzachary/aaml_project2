import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.autograd as autograd
import torch

outf = "./dcgan_results"

img_size = 128
latent_dim = 256
channels = 3
batch_size = 64
n_epochs = 200
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
sample_interval = 400
n_critic = 5
clip_value = 0.01

img_shape = (channels, img_size, img_size)

# Configure data loader
dataroot = "./nudes"

# folder dataset
dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
workers = 2
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(workers))

cuda = True if torch.cuda.is_available() else False

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from custom_layers import *
import copy


# defined for code simplicity.
def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, only=False):
    if wn:  layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad))
    else:   layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, gdrop=True, only=False):
    if gdrop:       layers.append(generalized_drop_out(mode='prop', strength=0.0))
    if wn:          layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad, initializer='kaiming'))
    else:           layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

def linear(layers, c_in, c_out, sig=True, wn=False):
    layers.append(Flatten())
    if wn:      layers.append(equalized_linear(c_in, c_out))
    else:       layers.append(Linear(c_in, c_out))
    if sig:     layers.append(nn.Sigmoid())
    return layers

    
def deepcopy_module(module, target):
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)                          # make new structure and,
            new_module[-1].load_state_dict(m.state_dict())         # copy weights
    return new_module

def soft_copy_param(target_link, source_link, tau):
    ''' soft-copy parameters of a link to another link. '''
    target_params = dict(target_link.named_parameters())
    for param_name, param in source_link.named_parameters():
        target_params[param_name].data = target_params[param_name].data.mul(1.0-tau)
        target_params[param_name].data = target_params[param_name].data.add(param.data.mul(tau))

def get_module_names(model):
    names = []
    for key, val in model.state_dict().iteritems():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_tanh = config.flag_tanh
        self.flag_norm_latent = config.flag_norm_latent
        self.nc = config.nc
        self.nz = config.nz
        self.ngf = config.ngf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_gen()

    def first_block(self):
        layers = []
        ndim = self.ngf
        if self.flag_norm_latent:
            layers.append(pixelwise_norm_layer())
        layers = deconv(layers, self.nz, ndim, 4, 1, 3, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        return  nn.Sequential(*layers), ndim

    def intermediate_block(self, resl):
        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2, resl)), int(pow(2, resl)))
        ndim = self.ngf
        if resl==3 or resl==4 or resl==5:
            halving = False
            ndim = self.ngf
        elif resl==6 or resl==7 or resl==8 or resl==9 or resl==10:
            halving = True
            for i in range(int(resl)-5):
                ndim = ndim/2
        ndim = int(ndim)
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))       # scale up by factor of 2.0
        if halving:
            layers = deconv(layers, ndim*2, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        else:
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        return  nn.Sequential(*layers), ndim, layer_name
    
    def to_rgb_block(self, c_in):
        layers = []
        layers = deconv(layers, c_in, self.nc, 1, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise, only=True)
        if self.flag_tanh:  layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def get_init_gen(self):
        model = nn.Sequential()
        first_block, ndim = self.first_block()
        model.add_module('first_block', first_block)
        model.add_module('to_rgb_block', self.to_rgb_block(ndim))
        self.module_names = get_module_names(model)
        return model
    
    def grow_network(self, resl):
        # we make new network since pytorch does not support remove_module()
        new_model = nn.Sequential()
        names = get_module_names(self.model)
        for name, module in self.model.named_children():
            if not name=='to_rgb_block':
                new_model.add_module(name, module)                      # make new structure and,
                new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights
            
        if resl >= 3 and resl <= 9:
            print('growing network[{}x{} to {}x{}]. It may take few seconds...'.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2,resl)), int(pow(2,resl))))
            low_resl_to_rgb = deepcopy_module(self.model, 'to_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_upsample', nn.Upsample(scale_factor=2, mode='nearest'))
            prev_block.add_module('low_resl_to_rgb', low_resl_to_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_block', inter_block)
            next_block.add_module('high_resl_to_rgb', self.to_rgb_block(ndim))

            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)
           
    def flush_network(self):
        try:
            print('flushing network... It may take few seconds...')
            # make deep copy and paste.
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_to_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_to_rgb')
           
            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                if name!='concat_block' and name!='fadein_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

            # now, add the high resolution block.
            new_model.add_module(self.layer_name, high_resl_block)
            new_model.add_module('to_rgb_block', high_resl_to_rgb)
            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model

    def freeze_layers(self):
        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print('freeze pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x.view(x.size(0), -1, 1, 1))
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_sigmoid = config.flag_sigmoid
        self.nz = config.nz
        self.nc = config.nc
        self.ndf = config.ndf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_dis()

    def last_block(self):
        # add minibatch_std_concat_layer later.
        ndim = self.ndf
        layers = []
        layers.append(minibatch_std_concat_layer())
        layers = conv(layers, ndim+1, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        layers = conv(layers, ndim, ndim, 4, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        layers = linear(layers, ndim, 1, sig=self.flag_sigmoid, wn=self.flag_wn)
        return  nn.Sequential(*layers), ndim
    
    def intermediate_block(self, resl):
        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2,resl)), int(pow(2,resl)), int(pow(2, resl-1)), int(pow(2, resl-1)))
        ndim = self.ndf
        if resl==3 or resl==4 or resl==5:
            halving = False
            ndim = self.ndf
        elif resl==6 or resl==7 or resl==8 or resl==9 or resl==10:
            halving = True
            for i in range(int(resl)-5):
                ndim = ndim/2
        ndim = int(ndim)
        layers = []
        if halving:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
            layers = conv(layers, ndim, ndim*2, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        else:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        
        layers.append(nn.AvgPool2d(kernel_size=2))       # scale up by factor of 2.0
        return  nn.Sequential(*layers), ndim, layer_name
    
    def from_rgb_block(self, ndim):
        layers = []
        layers = conv(layers, self.nc, ndim, 1, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        return  nn.Sequential(*layers)
    
    def get_init_dis(self):
        model = nn.Sequential()
        last_block, ndim = self.last_block()
        model.add_module('from_rgb_block', self.from_rgb_block(ndim))
        model.add_module('last_block', last_block)
        self.module_names = get_module_names(model)
        return model
    

    def grow_network(self, resl):
            
        if resl >= 3 and resl <= 9:
            print('growing network[{}x{} to {}x{}]. It may take few seconds...'.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2,resl)), int(pow(2,resl))))
            low_resl_from_rgb = deepcopy_module(self.model, 'from_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_downsample', nn.AvgPool2d(kernel_size=2))
            prev_block.add_module('low_resl_from_rgb', low_resl_from_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_from_rgb', self.from_rgb_block(ndim))
            next_block.add_module('high_resl_block', inter_block)

            new_model = nn.Sequential()
            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))

            # we make new network since pytorch does not support remove_module()
            names = get_module_names(self.model)
            for name, module in self.model.named_children():
                if not name=='from_rgb_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)

    def flush_network(self):
        try:
            print('flushing network... It may take few seconds...')
            # make deep copy and paste.
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_from_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_from_rgb')
           
            # add the high resolution block.
            new_model = nn.Sequential()
            new_model.add_module('from_rgb_block', high_resl_from_rgb)
            new_model.add_module(self.layer_name, high_resl_block)
            
            # add rest.
            for name, module in self.model.named_children():
                if name!='concat_block' and name!='fadein_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model
    
    def freeze_layers(self):
        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print('freeze pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x


k = 2
p = 6



#################Train#######################

class trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.nz = config.nz
        self.optimizer = config.optimizer

        self.resl = 2           # we start from 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.globalIter = 0
        self.globalTick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen':None, 'dis':None}
        self.complete = {'gen':0, 'dis':0}
        self.phase = 'init'
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift
        
        # network and cirterion
        self.G = net.Generator(config)
        self.D = net.Discriminator(config)
        print ('Generator structure: ')
        print(self.G.model)
        print ('Discriminator structure: ')
        print(self.D.model)
        self.mse = torch.nn.MSELoss()
        if self.use_cuda:
            self.mse = self.mse.cuda()
            torch.cuda.manual_seed(config.random_seed)
            if config.n_gpu==1:
                self.G = torch.nn.DataParallel(self.G).cuda(device=0)
                self.D = torch.nn.DataParallel(self.D).cuda(device=0)
            else:
                gpus = []
                for i  in range(config.n_gpu):
                    gpus.append(i)
                self.G = torch.nn.DataParallel(self.G, device_ids=gpus).cuda()
                self.D = torch.nn.DataParallel(self.D, device_ids=gpus).cuda()  

        
        # define tensors, ship model to cuda, and get dataloader.
        self.renew_everything()
        
        # tensorboard
        self.use_tb = config.use_tb
        if self.use_tb:
            self.tb = tensorboard.tf_recorder()
        

    def resl_scheduler(self):
        '''
        this function will schedule image resolution(self.resl) progressively.
        it should be called every iteration to ensure resl value is updated properly.
        step 1. (trns_tick) --> transition in generator.
        step 2. (stab_tick) --> stabilize.
        step 3. (trns_tick) --> transition in discriminator.
        step 4. (stab_tick) --> stabilize.
        '''
        if floor(self.resl) != 2 :
            self.trns_tick = self.config.trns_tick
            self.stab_tick = self.config.stab_tick
        
        self.batchsize = self.loader.batchsize
        delta = 1.0/(2*self.trns_tick+2*self.stab_tick)
        d_alpha = 1.0*self.batchsize/self.trns_tick/self.TICK

        # update alpha if fade-in layer exist.
        if self.fadein['gen'] is not None:
            if self.resl%1.0 < (self.trns_tick)*delta:
                self.fadein['gen'].update_alpha(d_alpha)
                self.complete['gen'] = self.fadein['gen'].alpha*100
                self.phase = 'gtrns'
            elif self.resl%1.0 >= (self.trns_tick)*delta and self.resl%1.0 < (self.trns_tick+self.stab_tick)*delta:
                self.phase = 'gstab'
        if self.fadein['dis'] is not None:
            if self.resl%1.0 >= (self.trns_tick+self.stab_tick)*delta and self.resl%1.0 < (self.stab_tick + self.trns_tick*2)*delta:
                self.fadein['dis'].update_alpha(d_alpha)
                self.complete['dis'] = self.fadein['dis'].alpha*100
                self.phase = 'dtrns'
            elif self.resl%1.0 >= (self.stab_tick + self.trns_tick*2)*delta and self.phase!='final':
                self.phase = 'dstab'
            
        prev_kimgs = self.kimgs
        self.kimgs = self.kimgs + self.batchsize
        if (self.kimgs%self.TICK) < (prev_kimgs%self.TICK):
            self.globalTick = self.globalTick + 1
            # increase linearly every tick, and grow network structure.
            prev_resl = floor(self.resl)
            self.resl = self.resl + delta
            self.resl = max(2, min(10.5, self.resl))        # clamping, range: 4 ~ 1024

            # flush network.
            if self.flag_flush_gen and self.resl%1.0 >= (self.trns_tick+self.stab_tick)*delta and prev_resl!=2:
                if self.fadein['gen'] is not None:
                    self.fadein['gen'].update_alpha(d_alpha)
                    self.complete['gen'] = self.fadein['gen'].alpha*100
                self.flag_flush_gen = False
                self.G.module.flush_network()   # flush G
                print(self.G.module.model)
                #self.Gs.module.flush_network()         # flush Gs
                self.fadein['gen'] = None
                self.complete['gen'] = 0.0
                self.phase = 'dtrns'
            elif self.flag_flush_dis and floor(self.resl) != prev_resl and prev_resl!=2:
                if self.fadein['dis'] is not None:
                    self.fadein['dis'].update_alpha(d_alpha)
                    self.complete['dis'] = self.fadein['dis'].alpha*100
                self.flag_flush_dis = False
                self.D.module.flush_network()   # flush and,
                print(self.D.module.model)
                self.fadein['dis'] = None
                self.complete['dis'] = 0.0
                if floor(self.resl) < self.max_resl and self.phase != 'final':
                    self.phase = 'gtrns'

            # grow network.
            if floor(self.resl) != prev_resl and floor(self.resl)<self.max_resl+1:
                self.lr = self.lr * float(self.config.lr_decay)
                self.G.grow_network(floor(self.resl))
                #self.Gs.grow_network(floor(self.resl))
                self.D.grow_network(floor(self.resl))
                self.renew_everything()
                self.fadein['gen'] = dict(self.G.model.named_children())['fadein_block']
                self.fadein['dis'] = dict(self.D.model.named_children())['fadein_block']
                self.flag_flush_gen = True
                self.flag_flush_dis = True

            if floor(self.resl) >= self.max_resl and self.resl%1.0 >= (self.stab_tick + self.trns_tick*2)*delta:
                self.phase = 'final'
                self.resl = self.max_resl + (self.stab_tick + self.trns_tick*2)*delta


            
    def renew_everything(self):
        # renew dataloader.
        # folder dataset
        dataset = dset.ImageFolder(root=dataroot,
                                       transform=transforms.Compose([
                                           transforms.Scale(img_size),
                                           transforms.CenterCrop(img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        workers = 2
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(workers))

        self.loader = dataloader
        self.loader.renew(min(floor(self.resl), self.max_resl))
        
        # define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.x_tilde = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.real_label = torch.FloatTensor(self.loader.batchsize).fill_(1)
        self.fake_label = torch.FloatTensor(self.loader.batchsize).fill_(0)
        
        # enable cuda
        if self.use_cuda:
            self.z = self.z.cuda()
            self.x = self.x.cuda()
            self.x_tilde = self.x.cuda()
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()
            torch.cuda.manual_seed(config.random_seed)

        # wrapping autograd Variable.
        self.x = Variable(self.x)
        self.x_tilde = Variable(self.x_tilde)
        self.z = Variable(self.z)
        self.real_label = Variable(self.real_label)
        self.fake_label = Variable(self.fake_label)
        
        # ship new model to cuda.
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
        
        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
        

    def feed_interpolated_input(self, x):
        if self.phase == 'gtrns' and floor(self.resl)>2 and floor(self.resl)<=self.max_resl:
            alpha = self.complete['gen']/100.0
            transform = transforms.Compose( [   transforms.ToPILImage(),
                                                transforms.Scale(size=int(pow(2,floor(self.resl)-1)), interpolation=0),      # 0: nearest
                                                transforms.Scale(size=int(pow(2,floor(self.resl))), interpolation=0),      # 0: nearest
                                                transforms.ToTensor(),
                                            ] )
            x_low = x.clone().add(1).mul(0.5)
            for i in range(x_low.size(0)):
                x_low[i] = transform(x_low[i]).mul(2).add(-1)
            x = torch.add(x.mul(alpha), x_low.mul(1-alpha)) # interpolated_x

        if self.use_cuda:
            return x.cuda()
        else:
            return x

    def add_noise(self, x):
        # TODO: support more method of adding noise.
        if self.flag_add_noise==False:
            return x

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde).item() * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
        return x + z

    def train(self):
        # noise for test.
        self.z_test = torch.FloatTensor(self.loader.batchsize, self.nz)
        if self.use_cuda:
            self.z_test = self.z_test.cuda()
        self.z_test = Variable(self.z_test, volatile=True)
        self.z_test.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
        
        for step in range(2, self.max_resl+1+5):
            for iter in tqdm(range(0,(self.trns_tick*2+self.stab_tick*2)*self.TICK, self.loader.batchsize)):
                self.globalIter = self.globalIter+1
                self.stack = self.stack + self.loader.batchsize
                if self.stack > ceil(len(self.loader.dataset)):
                    self.epoch = self.epoch + 1
                    self.stack = int(self.stack%(ceil(len(self.loader.dataset))))

                # reslolution scheduler.
                self.resl_scheduler()
                
                # zero gradients.
                self.G.zero_grad()
                self.D.zero_grad()

                # update discriminator.
                self.x.data = self.feed_interpolated_input(self.loader.get_batch())
                if self.flag_add_noise:
                    self.x = self.add_noise(self.x)
                self.z.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
                self.x_tilde = self.G(self.z)
               
                self.fx = self.D(self.x)
                self.fx_tilde = self.D(self.x_tilde.detach())
                
                loss_d = self.mse(self.fx.squeeze(), self.real_label) + \
                                  self.mse(self.fx_tilde, self.fake_label)
                loss_d.backward()
                self.opt_d.step()

                # update generator.
                fx_tilde = self.D(self.x_tilde)
                loss_g = self.mse(fx_tilde.squeeze(), self.real_label.detach())
                loss_g.backward()
                self.opt_g.step()
                
                # logging.
                log_msg = ' [E:{0}][T:{1}][{2:6}/{3:6}]  errD: {4:.4f} | errG: {5:.4f} | [lr:{11:.5f}][cur:{6:.3f}][resl:{7:4}][{8}][{9:.1f}%][{10:.1f}%]'.format(self.epoch, self.globalTick, self.stack, len(self.loader.dataset), loss_d.item(), loss_g.item(), self.resl, int(pow(2,floor(self.resl))), self.phase, self.complete['gen'], self.complete['dis'], self.lr)
                tqdm.write(log_msg)

                # save model.
                self.snapshot('repo/model')

                # save image grid.
                if self.globalIter%self.config.save_img_every == 0:
                    with torch.no_grad():
                        x_test = self.G(self.z_test)
                    utils.mkdir('repo/save/grid')
                    utils.save_image_grid(x_test.data, 'repo/save/grid/{}_{}_G{}_D{}.jpg'.format(int(self.globalIter/self.config.save_img_every), self.phase, self.complete['gen'], self.complete['dis']))
                    utils.mkdir('repo/save/resl_{}'.format(int(floor(self.resl))))
                    utils.save_image_single(x_test.data, 'repo/save/resl_{}/{}_{}_G{}_D{}.jpg'.format(int(floor(self.resl)),int(self.globalIter/self.config.save_img_every), self.phase, self.complete['gen'], self.complete['dis']))

                # tensorboard visualization.
                if self.use_tb:
                    with torch.no_grad():
                        x_test = self.G(self.z_test)
                    self.tb.add_scalar('data/loss_g', loss_g[0].item(), self.globalIter)
                    self.tb.add_scalar('data/loss_d', loss_d[0].item(), self.globalIter)
                    self.tb.add_scalar('tick/lr', self.lr, self.globalIter)
                    self.tb.add_scalar('tick/cur_resl', int(pow(2,floor(self.resl))), self.globalIter)
                    '''IMAGE GRID
                    self.tb.add_image_grid('grid/x_test', 4, utils.adjust_dyn_range(x_test.data.float(), [-1,1], [0,1]), self.globalIter)
                    self.tb.add_image_grid('grid/x_tilde', 4, utils.adjust_dyn_range(self.x_tilde.data.float(), [-1,1], [0,1]), self.globalIter)
                    self.tb.add_image_grid('grid/x_intp', 4, utils.adjust_dyn_range(self.x.data.float(), [-1,1], [0,1]), self.globalIter)
                    '''

    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl' : self.resl,
                'state_dict' : self.G.module.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl' : self.resl,
                'state_dict' : self.D.module.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state


    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl' : self.resl,
                'state_dict' : self.G.module.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl' : self.resl,
                'state_dict' : self.D.module.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state


    def snapshot(self, path):
        if not os.path.exists(path):
            if os.name == 'nt':
                os.system('mkdir {}'.format(path.replace('/', '\\')))
            else:
                os.system('mkdir -p {}'.format(path))
        # save every 100 tick if the network is in stab phase.
        ndis = 'dis_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.globalTick)
        ngen = 'gen_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.globalTick)
        if self.globalTick%50==0:
            if self.phase == 'gstab' or self.phase =='dstab' or self.phase == 'final':
                save_path = os.path.join(path, ndis)
                if not os.path.exists(save_path):
                    torch.save(self.get_state('dis'), save_path)
                    save_path = os.path.join(path, ngen)
                    torch.save(self.get_state('gen'), save_path)
                    print('[snapshot] model saved @ {}'.format(path))

if __name__ == '__main__':
    ## perform training.
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')
    torch.backends.cudnn.benchmark = True           # boost speed.
    trainer = trainer(config)
    trainer.train()
    
#    save_image(fake_imgs.data[:25], outf + '/wgan_div_%d.png' % epoch, nrow=5, normalize=True)

#    torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
#    torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))