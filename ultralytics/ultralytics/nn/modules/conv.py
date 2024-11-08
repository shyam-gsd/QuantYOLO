# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn


from abc import ABCMeta
from abc import abstractmethod

import brevitas.nn as qnn
import functorch


__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "QuantConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "QuantConv",
    "QuantConcat",
    "QuantUpsamplingNearest2d",
    "QuantUnpackTensors",
    "Uint8ActPerTensorPoT",
    "Int8ActPerTensorPoT",
    "Int8WeightPerChannelPoT",

)

from brevitas import config
from brevitas.core.function_wrapper import CeilSte
from brevitas.inject.enum import RestrictValueType

from brevitas.quant_tensor import QuantTensor, _unpack_quant_tensor

from torch import Tensor

from torch.nn import UpsamplingNearest2d
from torch.nn.functional import interpolate

import brevitas.quant as quant



class Uint8ActPerTensorPoT(quant.Uint8ActPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte

class Int8ActPerTensorPoT(quant.Int8ActPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte

class Int8WeightPerChannelPoT(quant.Int8WeightPerChannelFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class QuantUnpackTensors(torch.nn.Module):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__()

    def forward(self, x):
        x = _unpack_quant_tensor(x)
        # print(x[0][0,:,0,0])
        return x

class QuantConv(nn.Module):

    """Simplified RepConv module with Conv fusing."""
    default_act = qnn.QuantReLU(act_quant=Uint8ActPerTensorPoT,bit_width= 6,return_quant_tensor=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True ,weight_quant=None,act_quant=None,**kwargs):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.bit_width = kwargs.get('bit_width', 6)
        self.act_quant = act_quant

        self.weight_quant = weight_quant
        self.weight_bit_width = kwargs.get('weight_bit_width', 6)
        self.return_quant_tensor = kwargs.get('return_quant_tensor', True)

        self.conv = qnn.QuantConv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False,weight_quant=self.weight_quant,weight_bit_width=self.weight_bit_width,return_quant_tensor=self.return_quant_tensor)
        self.bn = nn.BatchNorm2d(c2)
        default_act = qnn.QuantReLU(act_quant=self.act_quant,bit_width=self.bit_width,return_quant_tensor=self.return_quant_tensor)  # default activation
        self.act = default_act if act is True else act if isinstance(act, nn.Module) else qnn.QuantIdentity(return_quant_tensor=self.return_quant_tensor,act_quant=self.act_quant,bit_width=self.bit_width)


    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""

        return self.act(self.bn(self.conv(x)))

    def toggle_quantize(self, quantize):
        if quantize:
            self.conv.weight_quant = self.weight_quant
            self.conv.weight_bit_width = self.weight_bit_width

            self.act.act_quant = self.act_quant
            self.act.bit_width = self.bit_width





    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))

class QuantConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = qnn.QuantReLU(act_quant=Uint8ActPerTensorPoT,bit_width= 6,return_quant_tensor=True)  # default activation


    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = qnn.QuantConvTranspose2d(c1, c2, k, s, p, bias=not bn,weight_quant=Int8WeightPerChannelPoT,weight_bit_width= 6,return_quant_tensor=True)
        self.bn = nn.BatchNorm2d(c2) if bn else qnn.QuantIdentity(act_quant=Int8ActPerTensorPoT,bit_width= 6,return_quant_tensor=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else qnn.QuantIdentity(act_quant=Int8ActPerTensorPoT,bit_width= 6,return_quant_tensor=True)

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class QuantConcat(nn.Module):

    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def toggle_quantize(self, quantize):
        pass

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)



class ExportMixin(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._export_mode = False
        self.export_debug_name = None
        self.export_handler = None
        self.export_input_debug = False
        self.export_output_debug = False

    @property
    @abstractmethod
    def requires_export_handler(self):
        pass

    @property
    def export_mode(self):
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        if value and config.JIT_ENABLED:
            raise RuntimeError(
                "Export mode with BREVITAS_JIT is currently not supported. Save the model' "
                "state_dict to a .pth, load it back with BREVITAS_JIT=0, and call export.")
        if value and self.training:
            raise RuntimeError("Can't enter export mode during training, only during inference")
        if value and self.requires_export_handler and self.export_handler is None:
            raise RuntimeError("Can't enable export mode on a layer without an export handler")
        elif value and not self.requires_export_handler and self.export_handler is None:
            return  # don't set export mode when it's not required and there is no handler
        elif value and not self._export_mode and self.export_handler is not None:
            self.export_handler.prepare_for_export(self)
            self.export_handler.attach_debug_info(self)
        elif not value and self.export_handler is not None:
            self.export_handler = None
        self._export_mode = value


class _CachedIO:

    def __init__(self, quant_tensor: QuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.quant_tensor = quant_tensor.set(value=None)
        else:
            self.quant_tensor = quant_tensor

    @property
    def scale(self):
        return self.quant_tensor.scale

    @property
    def zero_point(self):
        return self.quant_tensor.zero_point

    @property
    def bit_width(self):
        return self.quant_tensor.bit_width

    @property
    def signed(self):
        return self.quant_tensor.signed


class QuantLayerMixin(ExportMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            return_quant_tensor: bool,
            cache_inference_quant_inp: bool = False,
            cache_inference_quant_out: bool = False,
            cache_quant_io_metadata_only: bool = True):
        ExportMixin.__init__(self)
        self.accept_quant_tensor = True
        self.return_quant_tensor = return_quant_tensor
        self.cache_inference_quant_inp = cache_inference_quant_inp
        self.cache_inference_quant_out = cache_inference_quant_out
        self.cache_quant_io_metadata_only = cache_quant_io_metadata_only
        self._cached_inp = None
        self._cached_out = None

    @property
    @abstractmethod
    def channelwise_separable(self) -> bool:
        pass

    @property
    def is_quant_input_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self._cached_inp is not None:
            return self._cached_inp.signed
        else:
            return None

    def _set_global_is_quant_layer(self, value):
        config._IS_INSIDE_QUANT_LAYER = value

    def quant_input_scale(self):
        if self._cached_inp is not None:
            return self._cached_inp.scale
        else:
            return None

    def quant_input_zero_point(self):
        if self._cached_inp is not None:
            return self._cached_inp.zero_point
        else:
            return None

    def quant_input_bit_width(self):
        if self._cached_inp is not None:
            return self._cached_inp.bit_width
        else:
            return None

    @property
    def is_quant_output_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self._cached_out is not None:
            return self._cached_out.signed
        else:
            return None

    def quant_output_scale(self):
        if self._cached_out is not None:
            return self._cached_out.scale
        else:
            return None

    def quant_output_zero_point(self):
        if self._cached_out is not None:
            return self._cached_out.zero_point
        else:
            return None

    def quant_output_bit_width(self):
        if self._cached_out is not None:
            return self._cached_out.bit_width
        else:
            return None

    def unpack_input(self, inp: Union[Tensor, QuantTensor]):
        self._set_global_is_quant_layer(True)
        # Hack to recognize a QuantTensor that has decayed to a tuple
        # when used as input to tracing (e.g. during ONNX export)
        if (torch._C._get_tracing_state() is not None and isinstance(inp, tuple) and
                len(inp) == len(QuantTensor._fields) and all([isinstance(t, Tensor) for t in inp])):
            inp = QuantTensor(*inp)
        if isinstance(inp, QuantTensor):
            # don't cache values during export pass
            if not self.training and not self._export_mode and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
        else:
            inp = QuantTensor(inp, training=self.training)
            if not self.training and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
        # Remove any naming metadata to avoid dowmstream errors
        # Avoid inplace operations on the input in case of forward hooks
        if not torch._C._get_tracing_state():
            inp = inp.set(value=inp.value.rename(None))
        return inp

    def pack_output(self, quant_output: QuantTensor):
        if not self.training and self.cache_inference_quant_out:
            self._cached_out = _CachedIO(quant_output.detach(), self.cache_quant_io_metadata_only)
        self._set_global_is_quant_layer(False)
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.value


class QuantUpsamplingNearest2d(QuantLayerMixin, UpsamplingNearest2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool = True, **kwargs):
        UpsamplingNearest2d.__init__(self, size=size, scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def toggle_quantize(self, quantize):
        pass

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        y = x.set(value=y_value)
        return self.pack_output(y)

# class QuantUpsamplingNearest2d():
#     pass


