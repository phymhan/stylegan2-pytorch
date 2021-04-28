import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from models.stylegan_layers import EqualizedLinear
import numpy as np
import pdb
st = pdb.set_trace

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Reshape(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Reshape, self).__init__()

    def forward(self, input):
        return input.view(input.shape[0], -1)


class Squeeze(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Squeeze, self).__init__()

    def forward(self, input):
        return torch.squeeze(input)


class OneHot(nn.Module):
    def __init__(self, n_classes=10):
        super(OneHot, self).__init__()
        self.n_classes = n_classes

    def forward(self, labels):
        device = labels.device
        labels = F.one_hot(labels, num_classes=self.n_classes).float().to(device)
        return labels


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def get_last_layer(self):
        return [self.to_rgbs[-1].conv.weight, self.convs[-1].conv.weight]

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input, detach=False):
        shape = input.shape
        if shape[-1] > self.style_dim:
            style = self.style(input.view(-1, self.style_dim))
            style = style.view(*shape)
        else:
            style = self.style(input)
        return style.detach() if detach else style

    def get_styles(
        self,
        styles,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.get_latent(s) for s in styles]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t
        if len(styles) < 2:  # no mixing
            inject_index = self.n_latent
            if styles[0].ndim < 3:  # w is of dim [batch, 512], repeat at dim 1 for each block
                if styles[0].shape[1] == self.style_dim:
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                else:
                    latent = styles[0].view(styles[0].shape[0], -1, self.style_dim)
            else:  # w is of dim [batch, n_latent, 512]
                latent = styles[0]
        else:  # mixing
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)
        return latent

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        detach_style=False,
    ):
        if not input_is_latent:  # if `style' is z, then get w = self.style(z)
            styles = [self.get_latent(s, detach=detach_style) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:  # no mixing
            inject_index = self.n_latent

            if styles[0].ndim < 3:  # w is of dim [batch, 512], repeat at dim 1 for each block
                if styles[0].shape[1] == self.style_dim:
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                else:
                    latent = styles[0].view(styles[0].shape[0], -1, self.style_dim)

            else:  # w is of dim [batch, n_latent, 512]
                latent = styles[0]

        else:  # mixing
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)  # only batch_size of latent is used
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], architecture='resnet'):
        super().__init__()

        self.architecture = architecture

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        if architecture == 'resnet':
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=True, activate=False, bias=False
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.architecture == 'resnet':
            skip = self.skip(input)
            out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        size,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        in_channel=3,
        stddev_group=4,
        architecture='resnet',
        which_phi='lin2'
    ):
        """
        which_phi == 'vec': phi(x) is vectorized feature before final_linear
        which_phi == 'avg': phi(x) is AvgPooled feature before final_linear
        which_phi == 'lin': phi(x) is Linear(feature.view(-1))
        """
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.architecture = architecture
        self.which_phi = which_phi

        convs = [ConvLayer(in_channel, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(
                ResBlock(in_channel, out_channel, blur_kernel,
                    architecture=architecture,
                )
            )

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = stddev_group
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        if self.which_phi == 'lin1':
            self.final_linear = nn.Sequential(
                Reshape(),
                EqualLinear(channels[4] * 4 * 4, 1)
            )
        elif self.which_phi == 'lin2':
            self.final_linear = nn.Sequential(
                Reshape(),
                EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1)
            )
        elif self.which_phi == 'lin4':
            self.final_linear = nn.Sequential(
                Reshape(),
                EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1)
            )
        elif self.which_phi == 'avg1':
            self.final_linear = nn.Sequential(
                nn.AvgPool2d(4),
                Squeeze(),
                EqualLinear(channels[4], 1)
            )
        elif self.which_phi == 'avg2':
            self.final_linear = nn.Sequential(
                nn.AvgPool2d(4),
                Squeeze(),
                EqualLinear(channels[4], channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1)
            )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.final_linear(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        size,
        style_dim=512,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        which_latent='w_plus',
        which_phi='lin2',
        stddev_group=4,
        stddev_feat=1,
        reparameterization=False,
        return_tuple=True,  # backward compatibility
        latent_space='w',
        pca_state=None,
    ):
        """
        which_latent: 'w_plus' predict different w for all blocks; 'w_tied' predict
          a single w for all blocks; 'wb' predict w and b (bias) for all blocks;
          'wb_shared' predict shared w and different biases.
        """
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))
        self.n_latent = log_size * 2 - 2  # copied from Generator
        self.n_noises = (log_size - 2) * 2 + 1
        self.which_latent = which_latent
        self.which_phi = which_phi
        self.style_dim = style_dim
        if self.which_latent == 'w_plus':
            self.latent_full = style_dim * self.n_latent
        elif self.which_latent == 'w_tied':
            self.latent_full = style_dim
        else:
            raise NotImplementedError
        self.reparameterization = reparameterization
        self.return_tuple = return_tuple
        self.latent_space = latent_space
        self.register_buffer('pca_state', pca_state)
        assert((latent_space in ['w', 'p', 'pn', 'z']) or (pca_state is not None))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)

        self.stddev_group = stddev_group
        self.stddev_feat = stddev_feat

        self.final_conv = ConvLayer(in_channel + (self.stddev_group > 1), channels[4], 3)
        
        rep_mul = 2 if reparameterization else 1
        if self.which_phi == 'avg0':
            assert(channels[4] == self.latent_full * rep_mul)
            self.final_linear = nn.Sequential(
                nn.AvgPool2d(4),
                Squeeze(),
            )
        elif self.which_phi == 'avg1':
            self.final_linear = nn.Sequential(
                nn.AvgPool2d(4),
                Squeeze(),
                EqualLinear(channels[4], self.latent_full * rep_mul)
            )
        elif self.which_phi == 'lin1':
            self.final_linear = nn.Sequential(
                Reshape(),
                EqualLinear(channels[4] * 4 * 4, self.latent_full * rep_mul)
            )
        elif self.which_phi == 'lin2':
            self.final_linear = nn.Sequential(
                Reshape(),
                EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], self.latent_full * rep_mul)
            )
        else:
            raise NotImplementedError

    def latent_forward(self, style, latent_space, pca_state=None):
        if latent_space in ['w', 'z']:  # style is in W
            return style
        elif latent_space in ['p']:  # style is in P
            style = F.leaky_relu(style, 0.2)
            return style
        elif latent_space in ['pn']:  # style is in PN
            if style.shape[1] > self.style_dim:
                p = torch.cat([torch.matmul(w * pca_state['Lambda'], pca_state['CT']) + pca_state['mu']
                        for w in torch.split(style, self.style_dim, 1)], 1)
            else:
                p = torch.matmul(style * pca_state['Lambda'], pca_state['CT']) + pca_state['mu']
            style = F.leaky_relu(p, 0.2)
            return style

    def forward(self, input):
        out = self.convs(input)
        batch = out.shape[0]

        if self.stddev_group > 1:
            batch, channel, height, width = out.shape
            group = min(batch, self.stddev_group)
            stddev = out.view(
                group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
            )
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
            stddev = stddev.repeat(group, 1, height, width)
            out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.final_linear(out)

        if self.reparameterization:
            return out.chunk(2, dim=1)

        out = self.latent_forward(out, self.latent_space, self.pca_state)

        if self.return_tuple:
            return out, None
        return out


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=512, n_mlp=4, lr_mlp=0.1, hidden_dim=512, use_pixelnorm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        if use_pixelnorm:
            layers = [PixelNorm()]
        else:
            layers = []

        input_dim = latent_dim
        for i in range(n_mlp):
            output_dim = 1 if i == n_mlp-1 else hidden_dim
            layers.append(
                EqualLinear(
                    input_dim, output_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
            input_dim = output_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        # input should be of shape [N, latent_dim]
        if input.ndim > 2 or input.shape[1] > self.latent_dim:
            input = input.view(-1, self.latent_dim)
        out = self.layers(input)
        return out


class LinearResBlock(nn.Module):
    def __init__(self, latent_dim=512, n_mlp=2, lr_mlp=0.01, use_residual=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        layers = []
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_residual:
            return (input + out) / math.sqrt(2)
        else:
            return out


class LatentMLP(nn.Module):
    def __init__(self, latent_dim=512, n_mlp=8, lr_mlp=0.01, use_residual=False, use_pixelnorm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        n_layer_per_block = 2

        layers = [PixelNorm()] if use_pixelnorm else []

        for i in range(n_mlp//n_layer_per_block):
            layers.append(
                LinearResBlock(
                    latent_dim, n_layer_per_block, lr_mlp, use_residual
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        shape = input.shape
        if shape[-1] > self.latent_dim:
            out = self.layers(input.view(-1, self.latent_dim))
            out = out.view(*shape)
        else:
            out = self.layers(input)
        return out


class LatentPrior(nn.Module):
    def __init__(self, noise_dim=512, latent_dim=512, hidden_dim=512, n_mlp=8, lr_mlp=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        layers = [PixelNorm()]

        input_dim = noise_dim
        for i in range(n_mlp):
            output_dim = latent_dim if i == n_mlp-1 else hidden_dim
            layers.append(
                EqualLinear(
                    input_dim, output_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
            input_dim = output_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        return out


class FactorModule(nn.Module):
    def __init__(self, in_dim, out_dim, weight=None, n_head=1):
        """
        Divide input's in_dim into n_head parts, perform linear transformation
        and concat results in out_dim.
        """
        super().__init__()
        self.n_head = n_head
        self.in_dim_each = in_dim // n_head
        self.out_dim_each = out_dim // n_head
        self.weight = nn.ParameterList()
        if weight is None:
            for i in range(n_head):
                w = torch.randn(self.out_dim_each, self.in_dim_each)
                self.weight.append(nn.Parameter(w))
        else:
            if weight.ndim == 2:
                assert(weight.shape[0] == self.out_dim_each and weight.shape[1] == self.in_dim_each)
                for i in range(n_head):
                    self.weight.append(nn.Parameter(weight))
            else:
                for w in weight:
                    assert(w.shape[0] == self.out_dim_each and w.shape[1] == self.in_dim_each)
                    self.weight.append(nn.Parameter(w))
        # self.weight = nn.Parameter(weight)

    def forward(self, input):
        # input [N, T, D] or [N, D]
        inputs = torch.split(input, self.in_dim_each, input.ndim-1)
        outputs = []
        for i in range(self.n_head):
            outputs.append(F.linear(inputs[i], self.weight[i]))
        return torch.cat(outputs, input.ndim-1)
