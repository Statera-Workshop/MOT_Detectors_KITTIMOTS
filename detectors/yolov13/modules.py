"""
YOLOv13 building blocks - copied unchanged from the official code (https://github.com/iMoonLab/yolov13,
ultralytics/nn/modules/conv.py and block.py, AGPL-3.0): dependency-closed extraction of the blocks the
YOLOv13 YAML uses (HyperACE, FullPAD_Tunnel, DownsampleConv, DSC3k2, DSConv, A2C2f, ...).
[Unified] flash-attention is not installed: the official code then takes its standard attention path
(USE_FLASH_ATTN = False, same operation, no CUDA kernel dependency).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

flash_attn_func = None   # [Unified] unused: USE_FLASH_ATTN is False
logger = logging.getLogger("yolov13")

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
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class DSConv(nn.Module):
    """The Basic Depthwise Separable Convolution."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s,
            padding=p, dilation=d, groups=c_in, bias=bias
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


USE_FLASH_ATTN = False


class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)


class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


class DSBottleneck(nn.Module):
    """
    An improved bottleneck block using depthwise separable convolutions (DSConv).

    This class implements a lightweight bottleneck module that replaces standard convolutions with depthwise
    separable convolutions to reduce parameters and computational cost. 

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to use a residual shortcut connection. The connection is only added if c1 == c2. Defaults to True.
        e (float, optional): Expansion ratio for the intermediate channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv layer. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv layer. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv layer. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSBottleneck module.

    Examples:
        >>> import torch
        >>> model = DSBottleneck(c1=64, c2=64, shortcut=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)   
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)  
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    """
    An improved C3k module using DSBottleneck blocks for lightweight feature extraction.

    This class extends the C3 module by replacing its standard bottleneck blocks with DSBottleneck blocks,
    which use depthwise separable convolutions.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of DSBottleneck blocks to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections within the DSBottlenecks. Defaults to True.
        g (int, optional): Number of groups for grouped convolution (passed to parent C3). Defaults to 1.
        e (float, optional): Expansion ratio for the C3 module's hidden channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv in each DSBottleneck. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in each DSBottleneck. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv in each DSBottleneck. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k module (inherited from C3).

    Examples:
        >>> import torch
        >>> model = DSC3k(c1=128, c2=128, n=2, k1=3, k2=7)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 64, 64])
    """
    def __init__(
        self,
        c1,                
        c2,                 
        n=1,                
        shortcut=True,      
        g=1,                 
        e=0.5,              
        k1=3,               
        k2=5,               
        d2=1                 
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  

        self.m = nn.Sequential(
            *(
                DSBottleneck(
                    c_, c_,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        )


class DSC3k2(C2f):
    """
    An improved C3k2 module that uses lightweight depthwise separable convolution blocks.

    This class redesigns C3k2 module, replacing its internal processing blocks with either DSBottleneck
    or DSC3k modules.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of internal processing blocks to stack. Defaults to 1.
        dsc3k (bool, optional): If True, use DSC3k as the internal block. If False, use DSBottleneck. Defaults to False.
        e (float, optional): Expansion ratio for the C2f module's hidden channels. Defaults to 0.5.
        g (int, optional): Number of groups for grouped convolution (passed to parent C2f). Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections in the internal blocks. Defaults to True.
        k1 (int, optional): Kernel size for the first DSConv in internal blocks. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in internal blocks. Defaults to 7.
        d2 (int, optional): Dilation for the second DSConv in internal blocks. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k2 module (inherited from C2f).

    Examples:
        >>> import torch
        >>> # Using DSBottleneck as internal block
        >>> model1 = DSC3k2(c1=64, c2=64, n=2, dsc3k=False)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output1 = model1(x)
        >>> print(f"With DSBottleneck: {output1.shape}")
        With DSBottleneck: torch.Size([2, 64, 128, 128])
        >>> # Using DSC3k as internal block
        >>> model2 = DSC3k2(c1=64, c2=64, n=1, dsc3k=True)
        >>> output2 = model2(x)
        >>> print(f"With DSC3k: {output2.shape}")
        With DSC3k: torch.Size([2, 64, 128, 128])
    """
    def __init__(
        self,
        c1,          
        c2,         
        n=1,          
        dsc3k=False,  
        e=0.5,       
        g=1,        
        shortcut=True,
        k1=3,       
        k2=7,       
        d2=1         
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(
                    self.c, self.c,
                    n=2,           
                    shortcut=shortcut,
                    g=g,
                    e=1.0,  
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(
                    self.c, self.c,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )


class AdaHyperedgeGen(nn.Module):
    """
    Generates an adaptive hyperedge participation matrix from a set of vertex features.

    This module implements the Adaptive Hyperedge Generation mechanism. It generates dynamic hyperedge prototypes
    based on the global context of the input nodes and calculates a continuous participation matrix (A)
    that defines the relationship between each vertex and each hyperedge.

    Attributes:
        node_dim (int): The feature dimension of each input node.
        num_hyperedges (int): The number of hyperedges to generate.
        num_heads (int, optional): The number of attention heads for multi-head similarity calculation. Defaults to 4.
        dropout (float, optional): The dropout rate applied to the logits. Defaults to 0.1.
        context (str, optional): The type of global context to use ('mean', 'max', or 'both'). Defaults to "both".

    Methods:
        forward: Takes a batch of vertex features and returns the participation matrix A.

    Examples:
        >>> import torch
        >>> model = AdaHyperedgeGen(node_dim=64, num_hyperedges=16, num_heads=4)
        >>> x = torch.randn(2, 100, 64)  # (Batch, Num_Nodes, Node_Dim)
        >>> A = model(x)
        >>> print(A.shape)
        torch.Size([2, 100, 16])
    """
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)  
        elif context == "both":
            self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
    
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)          
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)          
        else:
            avg_context = X.mean(dim=1)           
            max_context, _ = X.max(dim=1)           
            context_cat = torch.cat([avg_context, max_context], dim=-1) 
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets           
        
        X_proj = self.pre_head_proj(X) 
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
        
        logits = self.dropout(logits)  

        return F.softmax(logits, dim=1)


class AdaHGConv(nn.Module):
    """
    Performs the adaptive hypergraph convolution.

    This module contains the two-stage message passing process of hypergraph convolution:
    1. Generates an adaptive participation matrix using AdaHyperedgeGen.
    2. Aggregates vertex features into hyperedge features (vertex-to-edge).
    3. Disseminates hyperedge features back to update vertex features (edge-to-vertex).
    A residual connection is added to the final output.

    Attributes:
        embed_dim (int): The feature dimension of the vertices.
        num_hyperedges (int, optional): The number of hyperedges for the internal generator. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the internal generator. Defaults to 4.
        dropout (float, optional): The dropout rate for the internal generator. Defaults to 0.1.
        context (str, optional): The context type for the internal generator. Defaults to "both".

    Methods:
        forward: Performs the adaptive hypergraph convolution on a batch of vertex features.

    Examples:
        >>> import torch
        >>> model = AdaHGConv(embed_dim=128, num_hyperedges=16, num_heads=8)
        >>> x = torch.randn(2, 256, 128) # (Batch, Num_Nodes, Dim)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 256, 128])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        
    def forward(self, X):
        A = self.edge_generator(X)  
        
        He = torch.bmm(A.transpose(1, 2), X) 
        He = self.edge_proj(He)
        
        X_new = torch.bmm(A, He)  
        X_new = self.node_proj(X_new)
        
        return X_new + X


class AdaHGComputation(nn.Module):
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    Examples:
        >>> import torch
        >>> model = AdaHGComputation(embed_dim=64, num_hyperedges=8, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32) # (B, C, H, W)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            context=context
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) 
        tokens = self.hgnn(tokens) 
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out 


class C3AH(nn.Module):
    """
    A CSP-style block integrating Adaptive Hypergraph Computation (C3AH).

    The input feature map is split into two paths.
    One path is processed by the AdaHGComputation module to model high-order correlations, while the other
    serves as a shortcut. The outputs are then concatenated to fuse features.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion ratio for the hidden channels. Defaults to 1.0.
        num_hyperedges (int, optional): The number of hyperedges for the internal AdaHGComputation. Defaults to 8.
        context (str, optional): The context type for the internal AdaHGComputation. Defaults to "both".

    Methods:
        forward: Performs a forward pass through the C3AH module.

    Examples:
        >>> import torch
        >>> model = C3AH(c1=64, c2=128, num_hyperedges=8)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])
    """
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)  
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_, 
                          num_hyperedges=num_hyperedges, 
                          num_heads=num_heads,
                          dropout=0.1,
                          context=context)
        self.cv3 = Conv(2 * c_, c2, 1)  
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FuseModule(nn.Module):
    """
    A module to fuse multi-scale features for the HyperACE block.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    Examples:
        >>> import torch
        >>> model = FuseModule(c_in=64, channel_adjust=False)
        >>> # Input is a list of features from different backbone stages
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out


class HyperACE(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=True):
        super().__init__()
        self.c = int(c2 * e1) 
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)
                    
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class DownsampleConv(nn.Module):
    """
    A simple downsampling block with optional channel adjustment.

    This module uses average pooling to reduce the spatial dimensions (H, W) by a factor of 2. It can
    optionally include a 1x1 convolution to adjust the number of channels, typically doubling them.

    Attributes:
        in_channels (int): The number of input channels.
        channel_adjust (bool, optional): If True, a 1x1 convolution doubles the channel dimension. Defaults to True.

    Methods:
        forward: Performs the downsampling and optional channel adjustment.

    Examples:
        >>> import torch
        >>> model = DownsampleConv(in_channels=64, channel_adjust=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """
    def __init__(self, in_channels, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if channel_adjust:
            self.channel_adjust = Conv(in_channels, in_channels * 2, 1)
        else:
            self.channel_adjust = nn.Identity() 

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))


class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out
