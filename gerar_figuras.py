"""Gera as figuras do tutorial a partir de dados/resultados reais do projeto.
Saída: pasta figuras/."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import cv2

os.makedirs("figuras", exist_ok=True)


def fig1_fluxograma():
    etapas = ["Webcam", "MediaPipe (landmarks do rosto)", "EAR (abertura do olho)",
              "CNN (aberto / fechado)", "Alerta de fadiga"]
    n = len(etapas)
    fig, ax = plt.subplots(figsize=(4.4, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, n * 2); ax.axis("off")
    for i, txt in enumerate(etapas):
        yc = (n - i) * 2 - 1
        ax.add_patch(FancyBboxPatch((1.5, yc - 0.55), 7, 1.1,
                     boxstyle="round,pad=0.1", fc="#dce8f7", ec="#2b5d8a", lw=1.6))
        ax.text(5, yc, txt, ha="center", va="center", fontsize=11)
        if i < n - 1:
            ax.annotate("", xy=(5, yc - 1.45), xytext=(5, yc - 0.55),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2b5d8a"))
    plt.savefig("figuras/fig1_fluxograma.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig3_olhos():
    ab = sorted(os.listdir("dataset/aberto"))[:4]
    fe = sorted(os.listdir("dataset/fechado"))[:4]
    fig, axs = plt.subplots(2, 4, figsize=(6, 3.2))
    for j, f in enumerate(ab):
        axs[0, j].imshow(cv2.cvtColor(cv2.imread(f"dataset/aberto/{f}"), cv2.COLOR_BGR2RGB))
        axs[0, j].axis("off")
    for j, f in enumerate(fe):
        axs[1, j].imshow(cv2.cvtColor(cv2.imread(f"dataset/fechado/{f}"), cv2.COLOR_BGR2RGB))
        axs[1, j].axis("off")
    fig.text(0.04, 0.72, "Aberto", rotation=90, va="center", fontsize=12)
    fig.text(0.04, 0.28, "Fechado", rotation=90, va="center", fontsize=12)
    plt.savefig("figuras/fig3_olhos.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig4_cnn():
    camadas = ["Entrada\n64x64x1", "Conv 16\n+ ReLU", "Pool\n32x32", "Conv 32\n+ ReLU",
               "Pool\n16x16", "Conv 64\n+ ReLU", "Pool\n8x8", "Dense 64", "Saída\n(2)"]
    n = len(camadas)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, n * 2); ax.set_ylim(0, 2); ax.axis("off")
    for i, txt in enumerate(camadas):
        xc = i * 2 + 1
        cor = "#e7f0e3" if ("Conv" in txt or "Dense" in txt) else \
              ("#f7e0e0" if "Saída" in txt else "#f3e8d8")
        ax.add_patch(FancyBboxPatch((xc - 0.85, 0.4), 1.7, 1.2,
                     boxstyle="round,pad=0.05", fc=cor, ec="#555", lw=1.2))
        ax.text(xc, 1.0, txt, ha="center", va="center", fontsize=8.5)
        if i < n - 1:
            ax.annotate("", xy=(xc + 1.15, 1.0), xytext=(xc + 0.85, 1.0),
                        arrowprops=dict(arrowstyle="-|>", color="#555"))
    plt.savefig("figuras/fig4_cnn.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig5_treinamento():
    # Valores reais registrados no treino (val_acc por epoca)
    cnn = [83.1, 92.6, 93.8, 95.7, 97.4, 97.7, 97.9, 98.0, 98.0, 97.9, 98.6, 98.4]
    vit = [58.9, 58.9, 58.9, 86.9, 86.4, 92.9, 93.9, 95.1, 94.2, 96.5, 96.6, 96.0]
    ep = list(range(1, 13))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, cnn, "o-", label="CNN", color="#2b7a2b", lw=2)
    ax.plot(ep, vit, "s--", label="ViT", color="#b8530a", lw=2)
    ax.set_xlabel("Época"); ax.set_ylabel("Acurácia de validação (%)")
    ax.set_ylim(50, 100); ax.grid(alpha=0.3); ax.legend()
    plt.savefig("figuras/fig5_treinamento.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig6_resultado():
    fig, ax = plt.subplots(figsize=(5, 4))
    barras = ax.bar(["CNN", "ViT"], [98.4, 96.0],
                    color=["#2b7a2b", "#b8530a"], width=0.5)
    ax.set_ylim(90, 100); ax.set_ylabel("Acurácia (%)")
    ax.set_title("Acurácia final na validação")
    for b, a in zip(barras, [98.4, 96.0]):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.1, f"{a}%",
                ha="center", fontsize=12)
    plt.savefig("figuras/fig6_resultado.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig1_fluxograma()
    fig3_olhos()
    fig4_cnn()
    fig5_treinamento()
    fig6_resultado()
    print("Figuras geradas em figuras/")
