# Driver Fatigue CNN — Detecção de Fadiga do Motorista

Projeto de **Visão Computacional** que utiliza **Redes Neurais Convolucionais (CNN)**
para classificação de imagens, tendo como **estudo de caso** a detecção de fadiga
do motorista em tempo real a partir de uma webcam.

> Tema do seminário: **Redes Neurais Convolucionais (CNN)**.
> A detecção de fadiga é a aplicação prática usada para demonstrar a tecnologia.

## Como funciona

O sistema combina duas camadas:

1. **Geométrica** — localiza o rosto e os olhos com o **MediaPipe**, calcula o
   **EAR** (*Eye Aspect Ratio*) e o **PERCLOS** (% de tempo com os olhos fechados).
2. **Aprendizado profundo** — uma **CNN** treinada classifica o estado do olho
   (aberto/fechado). O rótulo dos dados de treino é gerado automaticamente pelo EAR.

Ao detectar sinais de fadiga, o sistema emite um **alarme sonoro** e um aviso na tela.

## Estrutura

```
driver-fatigue-cnn/
├── detector_fadiga.py     # detector em tempo real (EAR + PERCLOS + alerta)
├── coletar_dados.py       # coleta e anotação automática do dataset
├── modelos.py             # arquitetura da CNN
├── treino.py              # treinamento da CNN
├── requirements.txt       # dependências
├── face_landmarker.task   # modelo de rosto (MediaPipe)
└── gesture_recognizer.task# modelo de gestos (MediaPipe)
```

## Como executar

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python detector_fadiga.py         # detector em tempo real
python coletar_dados.py           # coleta o dataset de olhos
python treino.py                  # treina a CNN
```

## Resultados

| Modelo | Acurácia | Parâmetros | Tempo/imagem |
|--------|----------|------------|--------------|
| **CNN** | **98,4%** | 285.634 | **0,43 ms** |

A CNN classifica o estado do olho (aberto/fechado) com **98,4%** de acurácia.

## Autor

Luisa Caetano Araújo — disciplina de Visão Computacional (IFMG Campus Formiga),
Prof. Me. Fernando Paim Lima.
