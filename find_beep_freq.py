"""
Encuentra la frecuencia del beep en archivos WAV.
Uso:
    python find_beep_freq.py "C:\\ruta\\a\\la\\carpeta\\con\\los\\beep"
    python find_beep_freq.py "C:\\ruta\\al\\archivo.wav"
Si no pasas ruta, busca *beep*.wav en la carpeta actual.
"""
import sys, glob, os, wave
import numpy as np


def peak_freqs(path):
    w = wave.open(path, "rb")
    sr = w.getframerate()
    sw = w.getsampwidth()
    ch = w.getnchannels()
    raw = w.readframes(w.getnframes())
    w.close()

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    data = np.frombuffer(raw, dtype=dtype).astype(float)
    if ch == 2:
        data = data[::2]  # stereo -> mono (canal izquierdo)

    if len(data) < sr // 10:
        return None

    # Buscar la ventana mas FUERTE (el beep es lo mas alto en un beep+silencio)
    win = int(sr * 0.20)            # ventana de 200ms
    if win >= len(data):
        seg = data
    else:
        # energia por bloques de 20ms para ubicar el beep
        step = int(sr * 0.02)
        best_i, best_e = 0, -1
        for i in range(0, len(data) - win, step):
            e = float(np.sum(data[i:i + win] ** 2))
            if e > best_e:
                best_e, best_i = e, i
        seg = data[best_i:best_i + win]

    seg = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)

    # Top 3 picos de frecuencia
    idx = np.argsort(spec)[::-1][:3]
    return sr, [(round(freqs[i]), round(float(spec[i]))) for i in idx]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "."
    if os.path.isfile(arg):
        files = [arg]
    else:
        files = sorted(glob.glob(os.path.join(arg, "*beep*.wav")))
        if not files:
            files = sorted(glob.glob(os.path.join(arg, "*.wav")))

    if not files:
        print(f"No encontre WAVs en: {arg}")
        return

    print(f"{'archivo':45} {'sr':>6}  picos (Hz : energia)")
    print("-" * 85)
    for f in files:
        try:
            res = peak_freqs(f)
            if res is None:
                print(f"{os.path.basename(f):45} (muy corto)")
                continue
            sr, picos = res
            picos_str = "   ".join(f"{hz}Hz:{en}" for hz, en in picos)
            print(f"{os.path.basename(f):45} {sr:>6}  {picos_str}")
        except Exception as e:
            print(f"{os.path.basename(f):45} ERROR: {e}")


if __name__ == "__main__":
    main()
