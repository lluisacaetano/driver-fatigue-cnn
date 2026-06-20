"""
Detector de fadiga do motorista em tempo real.
Usa o MediaPipe FaceLandmarker (API 'tasks') para achar os olhos/boca e calcula:
  - EAR (Eye Aspect Ratio): mede o quanto o olho esta aberto
  - PERCLOS: % do tempo com os olhos fechados
  - Bocejo (MAR) e contagem de piscadas

Pre-requisito: o arquivo 'face_landmarker.task' na mesma pasta.
Rode com:  python detector_fadiga.py   (ESC para sair)
"""

import sys
import time
import platform
import subprocess
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- Configuracoes ----------------
EAR_LIMIAR = 0.21        # abaixo disso o olho e considerado fechado
MAR_LIMIAR = 0.60        # acima disso a boca e considerada aberta (bocejo)
PERCLOS_LIMIAR = 0.25    # fracao de tempo com olhos fechados para alertar
JANELA_SEG = 15          # janela de tempo (segundos) usada no PERCLOS
FRAMES_BOCEJO = 20       # frames seguidos de boca aberta = 1 bocejo

# Indices dos landmarks do Face Mesh (validos tambem no FaceLandmarker)
OLHO_DIR = [33, 160, 158, 133, 153, 144]   # [canto, topo, topo, canto, base, base]
OLHO_ESQ = [362, 385, 387, 263, 373, 380]
BOCA_VERT = [13, 14]     # labio superior / inferior (interno)
BOCA_HORIZ = [78, 308]   # cantos da boca


def distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calcular_ear(pts):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
    vertical = distancia(pts[1], pts[5]) + distancia(pts[2], pts[4])
    horizontal = 2.0 * distancia(pts[0], pts[3])
    return vertical / horizontal


def calcular_mar(vert, horiz):
    """MAR = abertura vertical da boca / largura da boca."""
    return distancia(vert[0], vert[1]) / distancia(horiz[0], horiz[1])


def coords(landmarks, indices, w, h):
    """Converte landmarks normalizados (0-1) em pixels."""
    return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]


# Som do alarme (macOS). Outras opcoes em /System/Library/Sounds/
SOM_ALARME = "/System/Library/Sounds/Sosumi.aiff"
_alarme_proc = None


def tocar_alarme():
    """Toca o alarme sem travar o video. Re-dispara quando o som anterior acaba,
    criando um efeito de alarme continuo enquanto a fadiga persiste."""
    global _alarme_proc
    if _alarme_proc is not None and _alarme_proc.poll() is None:
        return  # ainda tocando
    try:
        if platform.system() == "Darwin":
            _alarme_proc = subprocess.Popen(["afplay", SOM_ALARME])
        else:
            print("\a", end="", flush=True)  # bip do terminal (outros sistemas)
    except Exception:
        pass


def garantir_permissao_camera():
    """No macOS, pede a permissao de camera (nativo) e ESPERA o usuario responder.
    Contorna o bug do OpenCV que falha sem aguardar o pop-up."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        from Foundation import NSRunLoop, NSDate, NSDefaultRunLoopMode
    except Exception:
        return  # nao e mac / sem pyobjc: segue e deixa o OpenCV tentar

    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    # 0 = nao determinado, 1 = restrito, 2 = negado, 3 = autorizado
    if status == 3:
        return
    if status in (1, 2):
        print("\n[!] Camera BLOQUEADA. Abra Ajustes do Sistema > Privacidade e "
              "Seguranca > Camera, ative o Terminal, de Cmd+Q e rode de novo.\n")
        return

    box = {}
    def handler(concedido):
        box["ok"] = bool(concedido)

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVMediaTypeVideo, handler)
    print("\n[i] Clique em OK no pop-up para liberar a camera...\n")
    loop = NSRunLoop.currentRunLoop()
    while "ok" not in box:
        loop.runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.1))


def criar_landmarker():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def criar_reconhecedor_gestos():
    base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.GestureRecognizer.create_from_options(options)


def main():
    # Sem argumento -> webcam (0). Com argumento -> caminho de um video.
    fonte = sys.argv[1] if len(sys.argv) > 1 else 0
    if fonte == 0:
        garantir_permissao_camera()   # forca o pop-up de permissao e espera

    landmarker = criar_landmarker()
    reconhecedor = criar_reconhecedor_gestos()
    cap = cv2.VideoCapture(fonte)

    historico = deque()          # (timestamp, olho_fechado?)
    contador_bocejo = 0
    total_bocejos = 0
    total_piscadas = 0
    olho_fechado_antes = False
    frame_id = 0
    frames_mao_aberta = 0
    FRAMES_PARA_SAIR = 35        # ~1,5s de mao aberta para encerrar

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)            # espelha 
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # API tasks: precisa de mp.Image e timestamp em ms crescente
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_id += 1
        resultado = landmarker.detect_for_video(mp_image, frame_id * 33)

        alerta = False

        if resultado.face_landmarks:
            lm = resultado.face_landmarks[0]

            # --- EAR (media dos dois olhos) ---
            ear = (calcular_ear(coords(lm, OLHO_DIR, w, h)) +
                   calcular_ear(coords(lm, OLHO_ESQ, w, h))) / 2.0

            # --- MAR (bocejo) ---
            mar = calcular_mar(coords(lm, BOCA_VERT, w, h),
                               coords(lm, BOCA_HORIZ, w, h))

            olho_fechado = ear < EAR_LIMIAR

            # Conta piscada na transicao fechado -> aberto
            if olho_fechado_antes and not olho_fechado:
                total_piscadas += 1
            olho_fechado_antes = olho_fechado

            # Conta bocejo (boca aberta por varios frames seguidos)
            if mar > MAR_LIMIAR:
                contador_bocejo += 1
            else:
                if contador_bocejo >= FRAMES_BOCEJO:
                    total_bocejos += 1
                contador_bocejo = 0

            # --- PERCLOS na janela de tempo ---
            agora = time.time()
            historico.append((agora, olho_fechado))
            while historico and agora - historico[0][0] > JANELA_SEG:
                historico.popleft()
            fechados = sum(1 for _, c in historico if c)
            perclos = fechados / len(historico) if historico else 0.0

            # --- Decisao de fadiga ---
            if perclos > PERCLOS_LIMIAR or contador_bocejo >= FRAMES_BOCEJO:
                alerta = True

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"PERCLOS: {perclos * 100:.0f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Piscadas: {total_piscadas}  Bocejos: {total_bocejos}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Rosto nao detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if alerta:
            tocar_alarme()
            cv2.putText(frame, "!! FADIGA DETECTADA !!", (10, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        # --- Gesto de MAO ABERTA para encerrar sozinho ---
        gesto = reconhecedor.recognize_for_video(mp_image, frame_id * 33)
        mao_aberta = bool(gesto.gestures) and \
            gesto.gestures[0][0].category_name == "Open_Palm"
        frames_mao_aberta = frames_mao_aberta + 1 if mao_aberta else 0
        if frames_mao_aberta > 0:
            pct = min(100, int(100 * frames_mao_aberta / FRAMES_PARA_SAIR))
            cv2.putText(frame, f"Mao aberta: saindo {pct}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        sair_por_gesto = frames_mao_aberta >= FRAMES_PARA_SAIR

        cv2.imshow("Deteccao de Fadiga do Motorista", frame)
        if (cv2.waitKey(1) & 0xFF == 27) or sair_por_gesto:   # ESC ou mao aberta
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
