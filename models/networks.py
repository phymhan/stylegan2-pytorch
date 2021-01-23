import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
import numpy as np
# from models.stylegan_layers import PixelNormLayer, EqualizedLinear
import random
import functools
import pdb

def get_networks(names, args):
    models = []
    for name in names:
        if name in ['D_I']:
            net = get_image_discriminator(args.model_netD_I, args)
        elif name in ['D_V']:
            net = get_video_discriminator(args.model_netD_V, args)
        elif name in ['D', 'D_c', 'D_head']:
            net = get_discriminator_head(args.model_netD, args)
        elif name in ['D_concat_z', 'D_head_concat_z']:
            net = get_discriminator_head_concat_z(args.model_netD, args)
        elif name in ['D_I_head']:
            net = get_image_discriminator_head(args.model_netD, args)
        elif name in ['G']:
            net = get_frame_generator(args.model_netG, args)
        elif name in ['G_I']:
            net = get_image_generator(args.model_netG_I, args)
        elif name in ['P']:
            net = get_recurrent_prior(args.model_netP, args)
        elif name in ['E']:
            net = get_frame_encoder(args.model_netE, args)
        elif name in ['E_I']:
            net = get_image_encoder(args.model_netE_I, args)
        elif name in ['Q']:
            net = get_inference_network(args.model_netQ, args)
        elif name in ['F']:
            net = get_mapping_network(args.model_netF, args)
        elif name in ['Pred', 'predictor']:
            net = get_predictor(args.model_netPred, args)
        elif name in ['MI']:
            net = get_mi_estimator(args.model_netMI, args)
        elif name in ['D_m']:
            net = get_sequence_discriminator(args.model_netD_m, args)
        elif name in ['C_c', 'classifier_content']:
            net = get_video_classifier(args.model_netC_c, args.num_classes_content, args)
        elif name in ['C_m', 'classifier_motion']:
            net = get_video_classifier(args.model_netC_m, args.num_classes_motion, args)
        elif name in ['feat']:
            net = get_perceptual_model(args.which_feat_model, args)
        elif name in ['F_I', 'feat_image']:
            net = get_perceptual_model(args.model_feat_I, args)
        elif name in ['F_V', 'feat_video']:
            net = get_perceptual_model_3D(args.model_feat_V, args)
        else:
            raise NotImplementedError
        models.append(net)
    return models


def get_frame_encoder(model, args):
    # Frame Encoder
    if model == 'empty':
        netE = EmptyModel()
    elif model == 'basic':
        netE = FrameEncoder(dim_out=args.dim_image_feature, ndf=args.dim_E_feature)
    elif model == 'dsvae':
        from models.dsvae_models import ConvLayers
        netE = ConvLayers(conv_dim=args.dim_image_feature, nf=256, in_size=args.image_size)
    elif model == 's3vae':
        from models.s3vae_models import encoder as DCGANEncoder
        netE = DCGANEncoder(args.dim_image_feature, nc=3, nf=args.dim_E_feature)
    elif model == 'tfgan':
        from models.tfgan_models import ResnetImageEncoder
        netE = ResnetImageEncoder(y_dim=0, size=args.image_size, output_dim=args.dim_image_feature)
        # print(f'Feature dimension of frame encoder is {netE.dim_feature}.')
        # assert(args.dim_image_feature == netE.dim_feature)
    elif model == 'biggan':
        from models.biggan_models import BigGANFrameEncoder
        netE = BigGANFrameEncoder(resolution=args.image_size, output_dim=args.dim_image_feature,
                                  D_ch=args.D_ch, D_attn=args.D_attn, D_init=args.D_init)
        # print(f'Feature dimension of frame encoder is {netE.dim_feature}.')
        # assert(args.dim_image_feature == netE.dim_feature)
    else:
        raise NotImplementedError
    return netE


def get_frame_generator(model, args):
    # Prepare dim_z
    if args.use_predictor:
        dim_z = args.dim_predictor_out if args.dim_predictor_out>0 else args.dim_z_content+args.dim_z_motion
        # TODO: Currently all decoders take concat z, so we just need: dim_z_motion + dim_z_content == dim_z
        dim_z_motion = 0
        dim_z_content = dim_z
    else:
        dim_z = args.dim_z_content + args.dim_z_motion
        dim_z_motion = args.dim_z_motion
        dim_z_content = args.dim_z_content
    use_rearrange_z = getattr(args, 'use_rearrange_z', False)
    if use_rearrange_z:
        assert(not args.use_predictor)
        dim_z = (args.dim_z_content, args.dim_z_motion)
    use_conditional = getattr(args, 'use_conditional', False)
    # Frame Generator
    if model == 'empty':
        netG = EmptyModel()
    elif model == 'basic':
        netG = FrameGenerator(n_channels=3, dim_z_content=dim_z_content, dim_z_motion=dim_z_motion,
                              ngf=args.dim_G_feature)
    elif model == 'tfgan':
        from models.tfgan_models import ResnetGenerator
        netG = ResnetGenerator(z_dim=dim_z, size=args.image_size, upsample=args.upsample)
    elif model == 'biggan':
        from models.biggan_models import Generator as BigGANGenerator
        netG = BigGANGenerator(dim_z=dim_z, resolution=args.image_size,
                               hier=args.hier, shared_dim=args.shared_dim,
                               G_ch=args.G_ch, G_attn=args.G_attn, G_init=args.G_init,
                               norm_style='bn', use_conditional=use_conditional,
                               use_rearrange_z=use_rearrange_z, G_shared=args.G_shared)
    elif model == 'dsvae':
        from models.dsvae_models import DeconvLayers
        netG = DeconvLayers(f_dim=dim_z_content, z_dim=dim_z_motion, conv_dim=args.dim_image_feature,
                            nf=256, in_size=args.image_size)
    elif model == 's3vae':
        from models.s3vae_models import decoder_conv
        netG = decoder_conv(dim_z, nc=3, nf=args.dim_G_feature)
    elif model == 's3vae_convt':
        from models.s3vae_models import decoder_convT
        netG = decoder_convT(dim_z, nc=3, nf=args.dim_G_feature)
    else:
        raise NotImplementedError
    return netG


def get_image_discriminator(model, args):
    # Image Discriminator
    if model == 'empty':
        netD_I = EmptyModel()
    elif model == 'basic':
        netD_I = ImageDiscriminator(n_channels=3)
    elif model == 'tfgan':
        from models.tfgan_models import ResnetImageDiscriminator
        netD_I = ResnetImageDiscriminator(y_dim=0, size=args.image_size)
    elif model == 'biggan':
        from models.biggan_models import Discriminator as BigGANDiscriminator
        netD_I = BigGANDiscriminator(resolution=args.image_size,
                                     D_ch=args.D_ch, D_attn=args.D_attn, D_init=args.D_init,
                                     use_conditional=getattr(args, 'use_conditional', False),
                                     use_D_feat_loss=getattr(args, 'use_D_feat_loss', False),
                                     D_feat_layer_idx=getattr(args, 'D_feat_layer_idx', 0))
    else:
        raise NotImplementedError
    return netD_I


def get_video_discriminator(model, args):
    # Video Discriminator
    if model == 'empty':
        netD_V = EmptyModel()
    elif model == 'basic':
        netD_V = VideoDiscriminator(n_channels=3)
    elif model == 'tfgan':
        from models.tfgan_models import ResnetVideoDiscriminator
        netD_V = ResnetVideoDiscriminator(y_dim=0, size=args.image_size)
    elif model == 'biggan':
        from models.biggan_models import Discriminator3D as BigGANDiscriminator3D
        netD_V = BigGANDiscriminator3D(resolution=args.image_size,
                                       D_ch=args.D_ch, D_attn=args.D_attn, D_init=args.D_init,
                                       use_conditional=False)
    else:
        raise NotImplementedError
    return netD_V


def get_recurrent_prior(model, args):
    # Recurrent Motion Network, or Learned Prior
    if model == 'empty':
        netP = EmptyModel()
    elif model == 'basic':
        netP = LearnedPrior(dim_z=args.dim_z_motion, teacher_forcing=not args.no_teacher_forcing,
                            scheduled_sampling_rate=args.scheduled_sampling_rate)
    elif model == 's3vae':
        from models.s3vae_models import S3VAERecurrentPrior
        netP = S3VAERecurrentPrior(args.dim_z_motion, args.dim_P_hidden, args.use_random_init,
                                   scheduled_sampling_rate=args.scheduled_sampling_rate,
                                   lstm_norm_type=args.lstm_norm_type)
    elif model == 'rwae':
        from models.rwae_models import RWAERecurrentPrior
        netP = RWAERecurrentPrior(args.dim_z_motion, args.dim_P_hidden,
                                  scheduled_sampling_rate=args.scheduled_sampling_rate)
    else:
        raise NotImplementedError
    return netP


def get_inference_network(model, args):
    # Inference model Q
    if model == 'empty':
        netQ = EmptyModel()
    elif model == 'basic':
        netQ = BasicInferenceModel(args.dim_z_content, args.dim_z_motion, args.dim_image_feature,
                                   args.dim_Q_hidden, args.num_layers_rnn)
    elif model == 'dsvae':
        from models.dsvae_models import InferenceModel as DSVAEPosterior
        netQ = DSVAEPosterior(f_dim=args.dim_z_content, z_dim=args.dim_z_motion,
                              conv_dim=args.dim_image_feature, hidden_dim=args.dim_Q_hidden,
                              bidirectional=not args.no_bidirectional, factorised=args.factorised,
                              condition_z_on_f=not args.no_condition_m_on_c)
    elif model == 's3vae':
        from models.s3vae_models import S3VAEPosterior
        netQ = S3VAEPosterior(args.dim_z_content, args.dim_z_motion, args.dim_image_feature,
                              args.dim_Q_hidden, not args.no_bidirectional)
    elif model == 'rwae':
        from models.rwae_models import RWAEPosterior
        netQ = RWAEPosterior(args.dim_z_content, args.dim_z_motion, args.dim_image_feature,
                             args.dim_Q_hidden)
    else:
        raise NotImplementedError
    return netQ


def get_predictor(model, args):
    dim_out = args.dim_predictor_out if args.dim_predictor_out>0 else args.dim_z_content+args.dim_z_motion
    # Predictor
    if model == 'empty':
        netPred = EmptyModel()
    elif model == 'rwae':
        from models.rwae_models import RWAEPredictor
        netPred = RWAEPredictor(args.dim_z_content + args.dim_z_motion, dim_out, args.dim_Pred_hidden)
    else:
        raise NotImplementedError
    return netPred


def get_discriminator_head(model, args):
    # Discriminator Head, by default takes z_content as input
    if model == 'empty':
        netD = EmptyModel()
    elif model == 'basic':
        netD = DiscriminatorHead(dim_in=args.dim_z_content, num_layers=args.num_layers_D,
                                 D_param=args.D_param, add_bn=args.D_add_bn)
    else:
        raise NotImplementedError
    return netD


def get_discriminator_head_concat_z(model, args):
    # Discriminator Head which takes concat z as input
    if model == 'empty':
        netD = EmptyModel()
    elif model == 'basic':
        netD = DiscriminatorHead(dim_in=args.dim_z_content + args.dim_z_motion,
                                 num_layers=args.num_layers_D,
                                 D_param=args.D_param, add_bn=args.D_add_bn)
    else:
        raise NotImplementedError
    return netD


def get_image_encoder(model, args):
    # Image Encoder, for WAE and ALAE
    if model == 'empty':
        netE = EmptyModel()
    elif model == 'basic':
        netE = FrameEncoder(dim_out=args.dim_z, ndf=args.dim_E_feature,
                            reparam=args.use_reparam_encoder)
    elif model == 's3vae':
        from models.s3vae_models import encoder as DCGANEncoder
        netE = DCGANEncoder(args.dim_z, 3, nf=args.dim_E_feature,
                            reparam=args.use_reparam_encoder)
    elif model == 'tfgan':
        from models.tfgan_models import ResnetImageEncoder
        netE = ResnetImageEncoder(y_dim=0, size=args.image_size, output_dim=args.dim_z,
                                  reparam=args.use_reparam_encoder)
    elif model == 'biggan':
        from models.biggan_models import BigGANFrameEncoder
        netE = BigGANFrameEncoder(resolution=args.image_size, output_dim=args.dim_z,
                                  D_ch=args.D_ch, D_attn=args.D_attn, D_init=args.D_init,
                                  reparam=args.use_reparam_encoder)
        # assert(args.dim_z == netE.dim_feature)
    else:
        raise NotImplementedError
    return netE


def get_image_discriminator_head(model, args):
    # Image Discriminator Head, for WAE and ALAE
    if model == 'empty':
        netD = EmptyModel()
    elif model == 'basic':
        netD = DiscriminatorHead(dim_in=args.dim_z, num_layers=args.num_layers_D,
                                 D_param=args.D_param, add_bn=args.D_add_bn)
    else:
        raise NotImplementedError
    return netD


def get_image_generator(model, args):
    # Image Generator, for WAE and ALAE
    if model == 'empty':
        netG = EmptyModel()
    elif model == 'basic':
        netG = FrameGenerator(n_channels=3, dim_z_content=args.dim_z, dim_z_motion=0, ngf=args.dim_G_feature)
    elif model == 'tfgan':
        from models.tfgan_models import ResnetGenerator
        netG = ResnetGenerator(z_dim=args.dim_z, size=args.image_size, upsample=args.upsample)
    elif model == 'biggan':
        from models.biggan_models import Generator as BigGANGenerator
        netG = BigGANGenerator(dim_z=args.dim_z, resolution=args.image_size,
                               hier=args.hier, shared_dim=args.shared_dim,
                               G_ch=args.G_ch, G_attn=args.G_attn, G_init=args.G_init,
                               norm_style='bn', use_conditional=getattr(args, 'use_conditional', False),
                               use_rearrange_z=args.use_rearrange_z, G_shared=args.G_shared)
    elif model == 's3vae':
        from models.s3vae_models import decoder_conv
        netG = decoder_conv(args.dim_z, nc=3, nf=args.dim_G_feature)
    elif model == 's3vae_convt':
        from models.s3vae_models import decoder_convT
        netG = decoder_convT(args.dim_z, nc=3, nf=args.dim_G_feature)
    else:
        raise NotImplementedError
    return netG


def get_mapping_network(model, args):
    # Mapping Network
    netF = Mapping(args.dim_z, args.dim_z, nf=args.dim_z, mapping_layers=args.n_layers_netF)
    return netF


def get_mi_estimator(model, args):
    # MI estimator using CLUB. For vCLUB, we predict z_content from z_motion.
    if model == 'empty':
        netMI = EmptyModel()
    elif model == 'mlp':
        from models.club import ClubMlpEstimator
        netMI = ClubMlpEstimator(args.dim_z_motion, args.dim_z_content, pooling=args.pooling_club,
                                 video_len=args.video_len)
    elif model == 'lstm':
        from models.club import ClubLstmEstimator
        netMI = ClubLstmEstimator(args.dim_z_motion, args.dim_z_content, hidden_size=args.dim_rnn_hidden,
                                  bidirectional=not args.no_bidirectional, n_layers=args.num_layers_rnn)
    else:
        raise NotImplementedError
    return netMI


def get_sequence_discriminator(model, args):
    # discriminator on z_motion
    if model == 'empty':
        netD = EmptyModel()
    elif model == 'mlp':
        from models.discriminators import MlpSeqeunceDiscriminator
        netD = MlpSeqeunceDiscriminator(args.dim_z_motion, pooling=args.pooling_club,
                                        video_len=args.video_len, n_layers=args.num_layers_D)
    elif model == 'lstm':
        from models.discriminators import LstmSeqeunceDiscriminator
        netD = LstmSeqeunceDiscriminator(args.dim_z_motion, hidden_size=args.dim_rnn_hidden,
                                         bidirectional=not args.no_bidirectional, n_layers=args.num_layers_D)
    else:
        raise NotImplementedError
    return netD


def get_video_classifier(model, num_classes, args):
    if model == 'empty':
        netF = EmptyModel()
    elif model == 'c3d_multi_8':
        from models.classifiers import MultiHeadClassifier8
        netC = MultiHeadClassifier8(n_channels=3, n_classes=num_classes)
    elif model == 'c3d_multi_16':
        from models.classifiers import MultiHeadClassifier16
        netC = MultiHeadClassifier16(n_channels=3, n_classes=num_classes)
    else:
        raise NotImplementedError
    return netC


def get_perceptual_model(model, args):
    if model == 'empty':
        netF = EmptyModel()
    elif model == 'VGG':
        from models.vgg_models import VGG16
        netF = VGG16(output_layer_idx=args.output_layer_idx)
        netF.load_state_dict(torch.load(args.weight_path_feat))
    else:
        raise NotImplementedError
    return netF


def get_perceptual_model_3D(model, args):
    # TODO: pretrained C3D mdoel
    return None


##################################################
# Helper classes and models for static images
##################################################
class EmptyModel(nn.Module):
    def __init__(self, **kwargs):
        super(EmptyModel, self).__init__()
        self.dummy_layer = nn.Linear(1, 1, False)


'''
A very basic MLP mapping network for ALAE, borrowed from:
https://github.com/huangzh13/StyleGAN.pytorch/blob/master/models/GAN.py
with some hard-coded hyper-parameters.
'''
class Mapping(nn.Module):
    def __init__(self, dim_in, dim_out, nf=128, mapping_layers=4):
        super(Mapping, self).__init__()
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}['lrelu']
        layers = []
        layers.append(('pixel_norm', PixelNormLayer()))
        layers.append(('dense0', EqualizedLinear(dim_in, nf, gain=gain, lrmul=0.01, use_wscale=True)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = nf
            fmaps_out = dim_out if layer_idx == mapping_layers - 1 else nf
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=0.01, use_wscale=True)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))
        self.map = nn.Sequential(OrderedDict(layers))
    
    def forward(self, z):
        h = self.map(z)
        return h


##################################################
# MoCoGAN Models (Legacy)
##################################################
class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * torch.randn_like(x)
        return x


class ImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(ImageDiscriminator, self).__init__()
        self.use_noise = use_noise
        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        h = self.main(x)
        return h


class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(PatchImageDiscriminator, self).__init__()
        self.use_noise = use_noise
        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        h = self.main(x)
        return h


class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()
        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma
        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        # x should be of shape [N, C, T, H, W]
        h = self.main(x)
        return h


class PatchVideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(PatchVideoDiscriminator, self).__init__()
        # minimum size: [N, C, 13, 16, 16]

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )

    def forward(self, x):
        # x should be of shape [N, C, T, H, W]
        h = self.main(x)
        return h


class FrameGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_motion, ngf=64):
        super(FrameGenerator, self).__init__()
        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        dim_z = dim_z_motion + dim_z_content
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)


class LearnedPrior(nn.Module):
    def __init__(self, dim_z, teacher_forcing=False, scheduled_sampling_rate=0):
        super(LearnedPrior, self).__init__()
        self.dim_z = dim_z
        self.teacher_forcing = teacher_forcing
        self.scheduled_sampling_rate = scheduled_sampling_rate
        if self.teacher_forcing:
            self.grucell = nn.GRUCell(self.dim_z + self.dim_z, self.dim_z)
        else:
            self.grucell = nn.GRUCell(self.dim_z, self.dim_z)
    
    def get_initial_state(self, N):
        return self.to_device(torch.randn(N, self.dim_z))
    
    def get_iteration_noise(self, N):
        return self.to_device(torch.randn(N, self.dim_z))
    
    def to_device(self, tensor):
        device = next(self.parameters()).device
        return tensor.cuda(device.index) if device.type == 'cuda' else tensor

    def forward(self, N, T, z_post=None, init_state=None):
        h0 = self.get_initial_state(N) if init_state is None else init_state
        h = []
        h_t = h0
        z0 = self.get_initial_state(N)
        z_prev = z0
        for t in range(T):
            z_t = self.get_iteration_noise(N)
            if self.teacher_forcing:
                z_t = torch.cat((z_t, z_prev), dim=1)
            h_t = self.grucell(z_t, h_t)
            # if self.teacher_forcing:
            #     z_prev = h_t if z_post is None else z_post[:, t, :]
            if self.teacher_forcing:
                z_prev = h_t if (random.random() < self.scheduled_sampling_rate or z_post is None) else z_post[:,t,:]
            h.append(h_t.unsqueeze(1))
        h = torch.cat(h, dim=1)
        return h, h, h  # return three copies, for compatibility purpose
    
    def cell(self, z, h):
        return self.grucell(z, h)


##################################################
# Basic models for MoCo-WAE, VAE, ALAE,
# that are borrowed from legacy MoCoGAN Models
##################################################
'''
Just a CNN feature extractor, borrowed from MoCoGAN ImageDiscriminator
'''
class FrameEncoder(nn.Module):
    def __init__(self, dim_out, ndf=64, use_noise=False, noise_sigma=None, reparam=False):
        super(FrameEncoder, self).__init__()
        self.use_noise = use_noise
        self.dim_out = dim_out
        self.reparam = reparam
        self.feature = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False)
        )
        self.head = nn.Conv2d(ndf * 8, dim_out, 1, 1, 0, bias=False)
        if self.reparam:
            self.head_logvar = nn.Conv2d(ndf * 8, dim_out, 1, 1, 0, bias=False)

    def reparameterize(self, mean, logvar, rsample=True):
        if not rsample:
            return mean
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5*logvar)
        z = mean + eps*std
        return z
    
    def forward(self, x):
        # x should be of shape [N*T, C, H, W]
        if x.dim() > 4:
            x = x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        h = self.feature(x)
        h = torch.sum(h, [2, 3], keepdim=True)
        out = self.head(h)
        if self.reparam:
            logvar = self.head_logvar(h)
            return out, logvar, self.reparameterize(out, logvar, self.training)
        return out


'''
Basic inference model inspired by that of S3VAE
feature --> netRNN_0 --> z_content
               +--> netRNN_1 --> z_motion
'''
class BasicInferenceModel(nn.Module):
    def __init__(self, dim_z_content, dim_z_motion, dim_in, hidden_size=256, num_layers=1):
        super(BasicInferenceModel, self).__init__()
        self.dim_input = dim_in
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.netRNN_0 = nn.LSTM(input_size=dim_in, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=False, batch_first=True)
        self.netRNN_1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1,
                                bidirectional=False, batch_first=True)
        self.netZ_content = ReparamHead(hidden_size, dim_z_content)
        self.netZ_motion = ReparamHead(hidden_size, dim_z_motion)

    def forward(self, f):
        # f: [N, T, D]
        rnn0_hid, _ = self.netRNN_0(f)  # rnn_hid: [N, T, D], h and c: [n_layer, N, D]
        rnn0_hid_last = rnn0_hid[:, -1, :]  # OR: rnn_hid_last = hc[0][-1, :, :]
        z_content_mean, z_content_logvar, z_content = self.netZ_content(rnn0_hid_last)
        rnn1_hid, _ = self.netRNN_1(rnn0_hid)
        z_motion_mean, z_motion_logvar, z_motion = self.netZ_motion(rnn1_hid)
        return (z_content_mean, z_content_logvar, z_content), (z_motion_mean, z_motion_logvar, z_motion)


class ReparamHead(nn.Module):
    def __init__(self, dim_in, dim_out, norm_type='none'):
        super(ReparamHead, self).__init__()
        if norm_type == 'sn':
            self.mean_head = nn.utils.spectral_norm(nn.Linear(dim_in, dim_out))
            self.logvar_head = nn.utils.spectral_norm(nn.Linear(dim_in, dim_out))
        elif norm_type == 'wn':
            self.mean_head = nn.utils.weight_norm(nn.Linear(dim_in, dim_out))
            self.logvar_head = nn.utils.weight_norm(nn.Linear(dim_in, dim_out))
        elif norm_type == 'ln':
            self.mean_head = nn.Sequential(nn.LayerNorm([dim_in]), nn.Linear(dim_in, dim_out))
            self.logvar_head = nn.Sequential(nn.LayerNorm([dim_in]), nn.Linear(dim_in, dim_out))
        else:
            self.mean_head = nn.Linear(dim_in, dim_out)
            self.logvar_head = nn.Linear(dim_in, dim_out)
    
    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mean + eps*std
        return z
    
    def forward(self, feature, rsample=None):
        if rsample is None:
            rsample = self.training
        z_mean = self.mean_head(feature)
        z_logvar = self.logvar_head(feature)
        if rsample:
            return z_mean, z_logvar, self.reparameterize(z_mean, z_logvar)
        else:
            return z_mean, z_logvar, z_mean


'''
MLP Discriminator head operates on latent embeddings
'''
class DiscriminatorHead(nn.Module):
    def __init__(self, dim_in, ndf=64, num_layers=3, D_param='basic', add_bn=True):
        super(DiscriminatorHead, self).__init__()
        dim_out = ndf if num_layers > 1 else 1
        self.D_param = D_param
        self.add_bn = add_bn
        if self.D_param.lower() in ['basic', 'none']:
            self.which_linear = nn.Linear
        elif self.D_param.lower() in ['sn']:
            import models.biggan_layers
            self.which_linear = models.biggan_layers.SNLinear
        layers = [self.which_linear(dim_in, dim_out)]
        if self.add_bn:
            for i in range(2, num_layers+1):
                dim_in = dim_out
                dim_out = ndf if i < num_layers else 1
                layers += [nn.BatchNorm1d(dim_in),
                           nn.LeakyReLU(0.2, inplace=True),
                           self.which_linear(dim_in, dim_out)]
        else:
            for i in range(2, num_layers+1):
                dim_in = dim_out
                dim_out = ndf if i < num_layers else 1
                layers += [nn.LeakyReLU(0.2, inplace=True),
                           self.which_linear(dim_in, dim_out)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, h):
        if h.dim() > 2:
            # h is of size [N, T, D]
            h = h.reshape(-1, h.size(-1))
        h = self.model(h)
        return h
