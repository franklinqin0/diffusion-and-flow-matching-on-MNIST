import math
from abc import ABC, abstractmethod
from typing import List, Type

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .simulators import ODE


class ConditionalVectorField(nn.Module, ABC):
    """
    Conditional vector field u_t^theta(x|y)
    """
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        pass


class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, null_label: int, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale
        self.null_label = null_label

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * self.null_label
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        super().__init__()
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        self.net = torch.nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPConditionalVectorField(ConditionalVectorField):
    def __init__(self, dim: int, hidden_dim: int, class_dim: int, num_classes: int):
        super().__init__()
        self.mlp = MLP([dim + class_dim + 1, hidden_dim, hidden_dim, dim])
        self.class_embedding = nn.Embedding(num_classes + 1, class_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        xyt = torch.cat([x, self.class_embedding(y), t.unsqueeze(-1)], dim=-1)
        return self.mlp(xyt)


class FourierEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        freqs = 2 * math.pi * self.weights * t
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        return torch.cat([cos_embed, sin_embed], dim=-1) * math.sqrt(2)


class Patchifier(nn.Module):
    def __init__(self, img_size: int, patch_size: int, dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.net = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b d h w -> b (h w) d"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Depatchifier(nn.Module):
    def __init__(self, img_size: int, patch_size: int, dim: int, final_dim: int = 10):
        super().__init__()
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        h = w = img_size // patch_size
        self.net = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            MLP([dim, 4 * dim, final_dim * patch_size ** 2]),
            Rearrange('b (h w) (f ph pw) -> b f (h ph) (w pw)', h=h, w=w, ph=patch_size, pw=patch_size, f=final_dim),
            nn.Conv2d(final_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MHA(nn.Module):
    """Multi-headed self-attention"""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.fold_heads = Rearrange('b n (h d) -> (b h) n d', h=heads)
        self.unfold_heads = Rearrange('(b h) n d -> b n (h d)', h=heads)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.fold_heads, (q, k, v))
        qk = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)
        x = torch.einsum('bij,bjd->bid', attn, v)
        x = self.unfold_heads(x)
        return self.out(x)


def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + bias


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6),
        )
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)
        self.attn = MHA(dim, heads)
        self.ff = MLP([dim, 4 * dim, dim])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = rearrange(self.ada_ln(c), 'b d -> b 1 d')
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)
        x = x + attn_gate * self.attn(modulate(self.norm1(x), attn_scale, attn_bias))
        x = x + ff_gate * self.ff(modulate(self.norm2(x), ff_scale, ff_bias))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, depth: int, n_tokens: int, dim: int, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            DiffusionTransformerLayer(dim=dim, **layer_kwargs)
            for _ in range(depth)
        ])
        self.pos_encodings = nn.Parameter(torch.randn(n_tokens, dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encodings.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, c)
        return x


class MNISTDiffusionTransformer(ConditionalVectorField):
    def __init__(self, patch_size: int = 8, num_layers: int = 12, dim: int = 256, heads: int = 4):
        super().__init__()
        self.time_embedder = FourierEncoder(dim)
        self.y_embedder = nn.Embedding(num_embeddings=11, embedding_dim=dim)

        img_size = 32
        self.patchifier = Patchifier(img_size, patch_size, dim)

        n_tokens = (32 // patch_size) ** 2
        self.dit = DiffusionTransformer(depth=num_layers, n_tokens=n_tokens, dim=dim, heads=heads)

        self.depatchifier = Depatchifier(img_size, patch_size, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embedder(t)
        y_embed = self.y_embedder(y)
        x = self.patchifier(x)
        x = self.dit(x, t_embed + y_embed)
        x = self.depatchifier(x)
        return x
