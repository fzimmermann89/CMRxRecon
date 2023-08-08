from typing import Union, Optional, Tuple, Callable, Literal, List
from functools import partial
import torch
import math
from torch import nn
from .blocks import *


class UnetLayer(nn.Module, LatEmbLayer):
    """
    One Layer in a Unet-based Net
    X--> Encoder -------------- Skip--------------- Decoder --> X"
          |-- downsampling -- SubLayers -- upsampling --|

    """

    def __init__(self, encoder, downsampling, sublayer, upsampling, decoder, skip=None):
        super().__init__()
        self.encoder = encoder
        self.skip = nn.Identity() if skip is None else skip
        self.decoder = decoder
        self.downpath = SequentialEmb(downsampling, sublayer, upsampling)

    def forward(self, x, emb, hin=None, hout=None):
        def call(module, x):
            if isinstance(module, LatEmbLayer):
                return module(x, emb, hin, hout)
            elif isinstance(module, EmbLayer):
                return module(x, emb)
            elif isinstance(module, LatLayer):
                return module(x, hin, hout)
            else:
                return module(x)

        x = call(self.encoder, x)
        xdown = call(self.downpath, x)
        skip = call(self.skip, x)
        x = call(self.decoder, (skip, xdown))
        return x


class Unet(nn.Module):
    def __init__(
        self,
        dim: Union[int, float],
        channels_in: int,
        channels_out: int,
        layer: int = 4,
        conv_per_enc_block: int = 2,
        conv_per_dec_block: int = 2,
        filters: int = 32,
        kernel_size: int = 3,
        norm: Union[bool, str, Callable[..., nn.Module]] = False,
        norm_before_activation: bool = True,
        bias: Union[bool, str] = True,
        dropout: float = 0.0,
        dropout_last: float = 0.0,
        padding_mode="zeros",
        residual=False,
        up_mode="linear",
        down_mode="maxpool",
        activation: Union[str, Callable[..., nn.Module]] = "relu",
        feature_growth: Union[Callable[[int], float], Tuple[float, ...], float] = 2.0,
        groups_enc: int = 1,
        groups_dec: int = 1,
        groups_last: int = 1,
        additional_last_decoder: int = 0,
        change_filters_last: bool = True,
        emb_dim: int = 0,
        skip_add: bool = False,
        downsample_dimensions: Optional[Tuple[Tuple[int, ...], ...]] = None,
        coordconv: Union[Literal["first"], bool, Tuple[Union[Tuple[bool, ...], bool]]] = False,
        latents=False,
    ):
        """
        A mostly vanilla UNet with linear final activation

        dim: 2 or 3 for 2D and 3D Unet, 2.5 for mostly-2D with an depth-wise conv in each block.
        channels_in: Number of Input channels
        channels_out: Number of Output channels
        layer: Number of layer, counted by down-steps
        conv_per_enc_block: Convolutions per encoder block
        conv_per_dec_block: Convolutions perdecoder block, not counting the upscaling
        filters: Initial number of filters
        kernel_size: size of Kernel in Encoder and Decoder Convolutions
        norm: False=No; True or 'batch'=BatchNorm; 'instance'=InstanceNorm
        norm_before_activation: Insert the norm before the activation, otherwise after the activation
        bias: True=Use bias in all convolutions; last=Use bias only in final 1-Conv
        dropout: Dropout in Encoder and Decoder. float=Dropout Propability or nn.Module. 0.=No dropout
        dropout_last: Dropout in final 1-Conv. float=Dropout Propability or nn.Module. 0.=No dropout
        padding_mode: padding mode for Convs or none.
        residual: Use residual connection between input and output. outer/inner/both=True/none=False
        up_mode: "conv"=Transposed Convolution: "nearest"/"linear"/"cubic": upsamling+Conv. if string contains '_reduce', channels will be reduced.
        down_mode: "conv"=Dilated Convolution, "maxpool", "averagepool" or "downshuffle"
        feature_growth: function depth:int->growth factor of feature number
        groups_enc/_dec/_last: Groups used in all encoder/decoder stages / last 1x1 Conv
        additional_last_decoder: Additional Convs used after last decoder before final output
        change_filters_last: Increase number of filters in last step of a block
        skip_add: Do an add of the skipped connection to the upsampled instead of a concat
        downsample_dimensions: Dimensions to downsample. None: 2D: (-1,-2), 3D: (-1,-2,-3), 2.5D: (-1,-2)
        coordconv: use coordconv as first conv. either bool or tuple[tuple[bool,...]|bool,tuple[bool,...]|bool] for each block of encoder/decoder
        latents: input / output hidden state used in skip connection mixing
        """
        super().__init__()

        if coordconv == "first":
            cc_enc, cc_dec = (True,) + (False,) * layer, False
        elif isinstance(coordconv, bool):
            cc_enc, cc_dec = coordconv, coordconv
        else:
            cc_enc, cc_dec = coordconv
        if isinstance(cc_enc, bool):
            cc_enc = (cc_enc,) * (layer + 1)
        if isinstance(cc_dec, bool):
            cc_dec = (cc_dec,) * layer
        cc_enc = cc_enc + (False,) * (layer + 1 - len(cc_enc))
        cc_dec = cc_dec + (False,) * (layer - len(cc_dec))
        coordconv = (cc_enc, cc_dec)

        if not (isinstance(residual, bool) or residual in ("outer", "inner", "both", "none")):
            raise ValueError("residual must be bool or 'outer'/'inner'/'both'/'none'")

        if isinstance(activation, str):
            activation = act(activation)

        if bias == "last":
            bias_last = True
            bias = False
        else:
            if not isinstance(bias, bool):
                raise ValueError("bias must be bool or 'last'")
            bias_last = bias

        if isinstance(feature_growth, (tuple, list)):
            _feature_growth = feature_growth
            feature_growth = lambda d: _feature_growth[d]
        elif isinstance(feature_growth, float):
            _feature_growth = feature_growth
            feature_growth = lambda d: _feature_growth

        fdim = dim
        dim = int(math.ceil(dim))
        half_dim = math.isclose(fdim - math.floor(fdim), 0.5)

        if isinstance(latents, (bool, int, str)):
            latents = (latents,) * (layer + 1)
        else:
            latents = latents + (False,) * (layer - len(latents) + 1)
        self.latent = False

        def latentlayer(lat, ch_in):
            if isinstance(lat, str):
                n = int(lat.split("_")[1]) if "_" in lat else ch_in
                if "mix" in lat.lower():
                    self.latent = True
                    return LatentMix(dim, ch_in, n)
                elif "film" in lat.lower():
                    self.latent = True
                    return LatentGU(dim, ch_in, n, film=True)
                elif "gu" in lat.lower():
                    self.latent = True
                    return LatentGU(dim, ch_in, n, film=False)
            elif isinstance(lat, bool) and lat:
                self.latent = True
                return LatentMix(dim, ch_in, ch_in)
            elif isinstance(lat, int) and lat:
                self.latent = True
                return LatentMix(dim, ch_in, lat)

            return None

        if downsample_dimensions is None:
            downsample_dimensions = (tuple(range(-dim + half_dim, 0)),)

        def calulate_window(dim: int, factor: float, active_dim: Tuple[int, ...]) -> Tuple[int | float, ...]:
            ret = [1] * dim
            for d in active_dim:
                ret[d] = factor
            return tuple(ret)

        _downsamplings: List[Callable] = []
        downsampling: Callable
        for d in downsample_dimensions:
            if down_mode == "maxpool":
                pooling_window = pooling_stride = calulate_window(dim, 2, d)
                downsampling = partial(lambda *_, **kw: MaxPoolNd(dim)(**kw), kernel_size=pooling_window, stride=pooling_stride)
            elif down_mode == "averagepool":
                pooling_window = pooling_stride = calulate_window(dim, 2, d)
                downsampling = partial(lambda *_, **kw: AvgPoolNd(dim)(**kw), kernel_size=pooling_window, stride=pooling_stride)
            elif down_mode == "conv":
                pooling_window = calulate_window(dim, kernel_size, d)
                pooling_stride = calulate_window(dim, 2, d)
                downsampling = partial(ConvNd(dim), stride=pooling_stride, kernel_size=pooling_window, padding="same")
            elif down_mode == "downshuffle":
                factor = calulate_window(dim, 2, d)
                downsampling = partial(DownShuffle(dim=dim, factor=factor))
            else:
                raise NotImplementedError(f"unknown down_mode {down_mode}")
            _downsamplings.append(downsampling)
        downsamplings = (_downsamplings + [_downsamplings[-1]] * (layer - len(_downsamplings)))[:layer]

        reduce_upsampling = "reduce" in up_mode

        if "_nofinal" in str(norm):
            norm = str(norm).replace("_nofinal", "")
            final_norm = False
        else:
            final_norm = True

        upm = up_mode.split("_")[0]
        _upsamplings: list[Callable] = []
        for d in downsample_dimensions:
            if upm == "conv":
                window = calulate_window(dim, kernel_size & ~1, d)
                stride = calulate_window(dim, 2, d)
                upsampling = partial(ConvTransposeNd(dim), kernel_size=window, stride=stride, bias=bias)
            elif upm in ("nearest", "linear", "cubic"):
                factor = calulate_window(dim, 2, d)
                window = calulate_window(dim, 3, d)
                upsampling = partial(
                    lambda in_channels, out_channels, **kw: Upsample(conv_channels=(in_channels, out_channels), **kw),
                    dim=dim,
                    factor=factor,
                    mode=upm,
                    keep_leading_dim=False,
                )

            else:
                raise NotImplementedError(f"unknown up_mode {up_mode}")
            _upsamplings.append(upsampling)
        upsamplings = (_upsamplings + [_upsamplings[-1]] * (layer - len(_upsamplings)))[:layer]

        block = partial(
            ResidualCBlock if residual in ("inner", True, "both") else CBlock,
            dim=dim,
            kernel_size=kernel_size,
            norm=norm,
            norm_before_activation=norm_before_activation,
            bias=bias,
            dropout=dropout,
            padding=padding_mode != "none",
            activation=activation,
            padding_mode="zeros" if padding_mode == "none" else padding_mode,
            split_dim=half_dim,
            final_norm=final_norm,
        )
        join = Concat("crop" if padding_mode == "none" else "pad")
        if change_filters_last:
            features_enc = [(channels_in,) + (filters,) * (conv_per_enc_block - 1) + (int(filters * feature_growth(0)) & ~1,)]
        else:
            features_enc = [(channels_in,) + (filters,) + (conv_per_enc_block - 1) * (int(filters * feature_growth(0)) & ~1,)]

        last = features_enc[-1][-1]
        if skip_add:
            factor = 0
            join = Add()
        elif reduce_upsampling:
            factor = 1
        else:
            factor = feature_growth(1)

        features_dec = [(last + int(factor * last),) + (last,) * conv_per_dec_block]

        for depth in range(1, layer + 1):
            new = int(feature_growth(depth) * last) & ~1
            if change_filters_last:
                features_enc.append((last,) * conv_per_enc_block + (new,))
            else:
                features_enc.append((last,) + conv_per_enc_block * (new,))
            last = features_enc[-1][-1]
            if not (reduce_upsampling or skip_add):
                factor = feature_growth(depth + 1)
            features_dec.append((last + int(factor * last) & ~1,) + (last,) * conv_per_dec_block)
        net = block(features_enc[-1], coordconv=coordconv[0][-1])
        if (latent := latentlayer(latents[-1], features_enc[-1][-1])) is not None:
            net = SequentialEmb(net, latent)
        for fenc, fdec, fup, fdown, downsampling, upsampling, coordconv_enc, coordconv_dec, lat in zip(
            features_enc[-2::-1],
            features_dec[-2::-1],
            features_enc[-1::-1],
            features_enc[-3::-1] + [[1]],
            downsamplings[-1::-1],
            upsamplings[-1::-1],
            coordconv[0][-2::-1],
            coordconv[1][-1::-1],
            latents[-2::-1],
        ):
            decoder = SequentialEmb(join, block(fdec, groups=groups_dec, emb_dim=emb_dim, coordconv=coordconv_dec))
            encoder = block(fenc, groups=groups_enc, emb_dim=emb_dim, coordconv=coordconv_enc)
            up = upsampling(fup[-1], fdec[0] if skip_add else fdec[0] - fenc[-1])
            down = downsampling(fenc[-1], fup[0])
            lat = latentlayer(lat, fenc[-1])
            net = UnetLayer(encoder, down, net, up, decoder, skip=lat)
        self.net = net

        self.last = CBlock(
            (features_enc[0][-1], channels_out),
            dim=dim,
            kernel_size=1,
            dropout=dropout_last,
            bias=bias_last,
            activation=None,
            groups=groups_last,
        )
        if additional_last_decoder > 0:
            self.last = (
                CBlock(
                    (features_enc[0][-1],) * (additional_last_decoder + 1),
                    dim=dim,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=padding_mode != "none",
                    activation=activation,
                    padding_mode="zeros" if padding_mode == "none" else padding_mode,
                    split_dim=half_dim,
                )
                + self.last
            )
        if residual in ("outer", True, "both"):
            self.residual = (
                nn.Identity()
                if channels_in == channels_out
                else ConvNd(dim)(channels_in, channels_out, kernel_size=1, bias=False)
            )
        else:
            self.residual = None
        self.emb_dim = emb_dim

        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        self.last[-1].bias.zero_()

    def forward(self, x, emb=None, hin=None):
        if self.emb_dim > 0 and emb is None:
            emb = torch.zeros(x.shape[0], self.emb_dim, dtype=x.dtype, device=x.device)
        hout = [] if self.latent else None
        ret = self.net(x, emb, hin, hout=hout)
        ret = self.last(ret)
        if self.residual is not None:
            ret = ret + self.residual(x)
        if self.latent:
            ret = (ret, hout)
        return ret
