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
import functools
import pdb
st = pdb.set_trace

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


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
            bias = None if self.bias is None else self.bias * self.lr_mul
            out = F.linear(
                input, self.weight * self.scale, bias=bias
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ConditionalEqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None,
        embed_dim=512,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim, embed_dim).div_(lr_mul))

        else:
            self.bias = None

        self.activation = activation

        self.scale_w = (1 / math.sqrt(in_dim)) * lr_mul
        self.scale_b = (1 / math.sqrt(embed_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input, labels):
        out = F.linear(input, self.weight * self.scale_w, bias=None)

        if self.bias is not None:
            bias = F.linear(labels, self.bias * self.scale_b, bias=None)
            out = out + bias * self.lr_mul

        if self.activation:
            out = fused_leaky_relu(out, bias=None)

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
        conditional_bias=False,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.conditional_bias = conditional_bias

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

        if self.conditional_bias:
            self.modulation = ConditionalEqualLinear(style_dim, in_channel, True, embed_dim=style_dim)
        else:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style, labels=None):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            if self.conditional_bias:
                style = self.modulation(style, labels)
            else:
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

        if self.conditional_bias:
            style = self.modulation(style, labels).view(batch, 1, in_channel, 1, 1)
        else:
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

    def forward(self, input, labels=None):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ConditionalInput(nn.Module):
    def __init__(self, channel, size=4, lr_mlp=0.01, embed_dim=512):
        super().__init__()

        self.channel = channel
        self.size = size
        self.input = EqualLinear(
            embed_dim, channel * size * size, lr_mul=lr_mlp,
            # activation="fused_lrelu",
        )

    def forward(self, input, labels):
        out = self.input(labels)

        return out.reshape(-1, self.channel, self.size, self.size)


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
        fused_bias_linear=None,  # fused_bias(embed) -> out_channel
        conditional_bias=False,  # conditional bias in modulation
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
            conditional_bias=conditional_bias,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.conditional_fused = fused_bias_linear is not None
        if self.conditional_fused:
            self.bias = fused_bias_linear(out_dim=out_channel)
            self.activate = FusedLeakyReLU(out_channel, bias=False)
        else:
            self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, labels=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        if self.conditional_fused:
            bias = self.bias(labels)
            rest_dim = [1] * (out.ndim - bias.ndim)
            out = out + bias.view(bias.shape[0], bias.shape[1], *rest_dim)
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


class OneHot(nn.Module):
    def __init__(self, n_classes=10):
        super(OneHot, self).__init__()
        self.n_classes = n_classes

    def forward(self, labels):
        device = labels.device
        labels = F.one_hot(labels, num_classes=self.n_classes).float().to(device)
        return labels


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        n_classes=10,
        conditional_strategy='ProjGAN',
        embed_is_linear=False,
        conditional_style_in=True,    # [z, y] --> w
        conditional_style_out=False,  # w + y --> w
        conditional_input=False,      # input(y)
        conditional_fused=False,      # bias(y) in fused leaky relu
        conditional_bias=False,       # style bias is conditional
        conditional_noise=False,      # make B conditional (channelwise)
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        self.n_classes = n_classes
        self.conditional_strategy = conditional_strategy
        self.embed_is_linear = embed_is_linear
        self.conditional_style_in = conditional_style_in
        self.conditional_style_out = conditional_style_out
        self.conditional_input = conditional_input
        self.conditional_fused = conditional_fused

        # Conditional embedding
        lr_emb = 1.
        self.embed_dim = style_dim
        if embed_is_linear:
            self.shared = nn.Sequential(
                OneHot(n_classes),
                EqualLinear(
                    n_classes, style_dim, lr_mul=lr_emb,
                    bias=False, activation=None,
                ),
            )
            
        else:
            self.shared = nn.Embedding(n_classes, style_dim)
        
        if self.conditional_fused:
            bias_linear = functools.partial(
                EqualLinear,
                in_dim=self.embed_dim,
                bias=False,
                lr_mul=lr_emb,
                activation=None,
            )
        else:
            bias_linear = None
        
        self.pixel_norm = PixelNorm()

        if self.conditional_style_in:
            layers = [EqualLinear(style_dim + self.embed_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")]
            n_mlp -= 1
        else:
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

        if self.conditional_input:
            self.input = ConditionalInput(self.channels[4], 4, lr_mlp, self.embed_dim)
        else:
            self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel,
            bias_linear=bias_linear,
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
                    bias_linear=bias_linear,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel,
                    bias_linear=bias_linear
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

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

    def forward(
        self,
        styles,
        labels=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        label_is_embeding=False,
    ):
        if not label_is_embeding:
            labels = self.shared(labels)

        if not input_is_latent:
            # styles is noise
            if self.conditional_style_in:
                styles = [self.pixel_norm(s) for s in styles]
                # labels = self.pixel_norm(labels)
                styles = [torch.cat([s, labels], dim=1) for s in styles]
            styles = [self.style(s) for s in styles]

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
        
        if self.conditional_style_out:
            latent = latent + labels.unsqueeze(1).repeat(1, self.n_latent, 1)

        out = self.input(latent, labels)  # only batch_size of latent is used
        out = self.conv1(out, latent[:, 0], noise=noise[0], labels=labels)

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1, labels=labels)
            out = conv2(out, latent[:, i + 1], noise=noise2, labels=labels)
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
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

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
        n_classes=10,
        conditional_strategy='ProjGAN',
        embed_is_linear=False,
        which_phi='vec',
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

        self.n_classes = n_classes
        self.conditional_strategy = conditional_strategy
        self.embed_is_linear = embed_is_linear
        self.which_phi = which_phi
        if self.which_phi == 'vec':
            self.embed_dim = channels[4] * 4 * 4
        elif self.which_phi in ['avg', 'lin1', 'lin2']:
            self.embed_dim = channels[4]
        if embed_is_linear:
            self.embedding = EqualLinear(
                self.embed_dim, n_classes,
                bias=False, activation=None,
            )
        else:
            self.embedding = nn.Embedding(n_classes, self.embed_dim)
        self.pixel_norm = PixelNorm()

        convs = [ConvLayer(in_channel, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        if self.which_phi == 'vec':
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1),
            )  # final_linear is not linear
        elif self.which_phi == 'avg':
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4], channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1))
        elif self.which_phi == 'lin1':
            self.final_linear0 = EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu")
            self.final_linear = EqualLinear(channels[4], 1)
        elif self.which_phi == 'lin2':
            self.final_linear0 = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], channels[4]),
            )
            self.final_linear = EqualLinear(channels[4], 1)

        if self.conditional_strategy == 'InnerProd':
            self.final_linear = None

    def forward(self, input, labels=None):
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

        if self.which_phi == 'vec':  # h = phi(x), dim is 8192
            h = out.view(batch, -1)
        elif self.which_phi == 'avg':  # h = phi(x), dim is 512
            h = torch.sum(out, [2, 3]) / 16.
        elif self.which_phi in ['lin1', 'lin2']:  # h = phi(x), dim is 512
            h = self.final_linear0(out.view(batch, -1))

        if self.conditional_strategy == 'ProjGAN':
            out = torch.squeeze(self.final_linear(h))
        elif self.conditional_strategy == 'InnerProd':
            out = 0.

        if self.embed_is_linear:
            proj = self.embedding(h)[range(batch), labels]
        else:
            proj = torch.sum(torch.mul(self.embedding(labels), h), 1)
        
        return proj + out


class LinearResBlock(nn.Module):
    def __init__(self, latent_dim=512, n_mlp=2, use_residual=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        layers = []
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation="fused_lrelu"
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
    def __init__(self, latent_dim=512, n_mlp=8, use_residual=False, use_pixelnorm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        n_layer_per_block = 2
        if use_pixelnorm:
            layers = [PixelNorm()]
        else:
            layers = []

        for i in range(n_mlp//n_layer_per_block):
            layers.append(
                LinearResBlock(
                    latent_dim, n_layer_per_block, use_residual
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


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy = nn.Linear(1, 1, False)

    def forward(self, input):
        return input


class Encoder(nn.Module):
    def __init__(
        self,
        size,
        style_dim=512,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        which_latent='w_plus',
        reshape_latent=False,
        stddev_group=4,
        stddev_feat=1,
        reparameterization=False,
        return_tuple=True,  # backward compatibility
    ):
        """
        which_latent: 'w_plus' predict different w for all blocks; 'w_shared' predict
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
        self.reshape_latent = reshape_latent
        self.style_dim = style_dim
        self.reparameterization = reparameterization
        self.return_tuple = return_tuple

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = stddev_group
        self.stddev_feat = stddev_feat

        self.final_conv = ConvLayer(in_channel + (self.stddev_group > 1), channels[4], 3)
        if self.which_latent == 'w_plus':
            out_channel = style_dim * self.n_latent
        elif self.which_latent == 'w_shared':
            out_channel = style_dim
        else:
            raise NotImplementedError
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], out_channel),
        )
        if reparameterization:
            self.final_logvar = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], out_channel),
        )

    def forward(self, input, reshape=False):
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

        out = out.view(batch, -1)
        out_mean = self.final_linear(out)
        if self.reparameterization:
            out_logvar = self.final_logvar(out)
            return out_mean, out_logvar
        if self.which_latent == 'w_plus' and reshape:
            out_mean = out_mean.reshape(batch, self.n_latent, self.style_dim)
        if self.return_tuple:
            return out_mean, None
        return out_mean
