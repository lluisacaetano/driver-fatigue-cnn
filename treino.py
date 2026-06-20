"""
Treina a CNN e o ViT no dataset de olhos (aberto/fechado), avalia os dois e
imprime uma TABELA COMPARATIVA (acuracia, nº de parametros, velocidade).

Pre-requisito: rodar 'python coletar_dados.py' antes, para gerar dataset/.
Rode:  python treino.py
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from modelos import CNN, ViT

PASTA = "dataset"
EPOCAS = 12
BATCH = 64
LR = 1e-3
DEVICE = "cpu"   # CPU e suficiente (modelos pequenos) e 100% estavel


def carregar_dados():
    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(PASTA, transform=tf)
    n_val = max(1, int(0.2 * len(ds)))
    n_tr = len(ds) - n_val
    tr, val = random_split(ds, [n_tr, n_val],
                           generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr, batch_size=BATCH, shuffle=True),
            DataLoader(val, batch_size=BATCH), ds.classes, len(ds))


def avaliar(modelo, val):
    modelo.eval()
    certos = total = 0
    with torch.no_grad():
        for x, y in val:
            x, y = x.to(DEVICE), y.to(DEVICE)
            certos += (modelo(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return certos / total


def treinar(modelo, tr, val, nome):
    modelo.to(DEVICE)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(EPOCAS):
        modelo.train()
        soma = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(modelo(x), y)
            loss.backward()
            opt.step()
            soma += loss.item()
        acc = avaliar(modelo, val)
        print(f"  [{nome}] epoca {ep+1:>2}/{EPOCAS}  loss={soma/len(tr):.3f}  val_acc={acc*100:.1f}%")
    return acc


def velocidade_ms(modelo, n=200):
    """Tempo medio de inferencia por imagem (ms)."""
    modelo.eval()
    x = torch.randn(1, 1, 64, 64).to(DEVICE)
    with torch.no_grad():
        for _ in range(10):
            modelo(x)                      # aquecimento
        t0 = time.time()
        for _ in range(n):
            modelo(x)
        return (time.time() - t0) / n * 1000


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    # precisa ter coletado dados antes
    for c in ("aberto", "fechado"):
        p = os.path.join(PASTA, c)
        if not os.path.isdir(p) or len(os.listdir(p)) < 10:
            print(f"[!] Poucas imagens em '{p}'. "
                  f"Rode 'python coletar_dados.py' primeiro.")
            return

    tr, val, classes, n = carregar_dados()
    print(f"Dataset: {n} imagens | classes: {classes}\n")

    resultados = []
    for nome, modelo in (("CNN", CNN()), ("ViT", ViT())):
        print(f"=== Treinando {nome} ===")
        acc = treinar(modelo, tr, val, nome)
        ms = velocidade_ms(modelo)
        params = n_params(modelo)
        torch.save(modelo.state_dict(), f"{nome.lower()}.pt")
        resultados.append((nome, acc, params, ms))
        print()

    # Tabela final 
    print("================ COMPARACAO  CNN  vs  ViT ================")
    print(f"{'Modelo':<8}{'Acuracia':<12}{'Parametros':<14}{'Tempo/img':<10}")
    print("-" * 44)
    for nome, acc, params, ms in resultados:
        print(f"{nome:<8}{acc*100:>6.1f}%     {params:>10,}    {ms:>6.2f} ms")

    with open("resultados.txt", "w") as f:
        f.write("Modelo,Acuracia(%),Parametros,Tempo_ms_por_img\n")
        for nome, acc, params, ms in resultados:
            f.write(f"{nome},{acc*100:.1f},{params},{ms:.2f}\n")
    print("\nModelos salvos (cnn.pt / vit.pt) e tabela em resultados.txt")


if __name__ == "__main__":
    main()
