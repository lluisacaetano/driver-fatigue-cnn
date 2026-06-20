"""
As duas arquiteturas que vamos comparar para classificar o estado do olho
(aberto x fechado), recebendo um recorte em tons de cinza de 64x64.

  - CNN: a Rede Neural Convolucional (protagonista do tema)
  - ViT: o Vision Transformer compacto (apenas para comparacao)
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN classica: filtros convolucionais que veem bordas -> formas -> olho."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # -> 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ViT(nn.Module):
    """Vision Transformer compacto, feito do zero.
    Corta a imagem em patches, trata como 'palavras' e usa auto-atencao."""

    def __init__(self, n_classes=2, img=64, patch=8, dim=64, depth=4, heads=4):
        super().__init__()
        self.n_patches = (img // patch) ** 2
        # 'patch embedding': cada patch vira um vetor de tamanho 'dim'
        self.proj = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))            # token de classe
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))  # posicao
        camada = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(camada, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        b = x.size(0)
        x = self.proj(x)                     # (b, dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)     # (b, n_patches, dim)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)       # acrescenta o token de classe
        x = x + self.pos                     # soma a posicao
        x = self.encoder(x)                  # auto-atencao
        return self.head(self.norm(x[:, 0]))  # usa o token de classe p/ decidir
