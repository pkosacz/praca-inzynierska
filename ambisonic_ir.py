from pathlib import Path
from scipy.signal import max_len_seq

import numpy as np
import soundfile as sf
import sounddevice as sd
import pyfar as pf
import datetime as dt

def generate_ess(f1=20.0, f2=20000.0, duration=10.0, fs=48000):
    """
    Generacja sygnału Exponential Sine Sweep (ESS)
    f1-f2: zakres częstotliwości
    duration: czas trwania sygnału
    fs: częstotliwość próbkowania
    """
    n_samples = int(fs * duration)

    start_margin = int(0.05 * n_samples)
    stop_margin  = int(0.05 * n_samples)

    ess_signal = pf.signals.exponential_sweep_freq(n_samples, [f1, f2], start_margin, stop_margin, sampling_rate=fs)

    return ess_signal

def generate_mls(order=10, fs=48000):
    """
    Generacja sygnału Maximum Length Sequence (MLS) i konwersja do wartości {-1, +1}
    order: liczba bitów użyta do generacji sygnału (2**n - 1)
    fs: częstotliwość próbkowania
    """
    bin_seq = max_len_seq(order)[0]

    y = 2.0 * bin_seq.astype(np.float32) - 1.0

    mls_signal = pf.Signal(y, fs)

    return mls_signal

def signal_to_audio_array(signal):
    """
    Konwersja z pf.Signal na format audio
    (n_channels, n_samples) -> (n_samples, n_channels)
    """
    x = np.array(signal.time, dtype=np.float32)  # (n_channels, n_samples)
    return np.transpose(x)  # (n_samples, n_channels)

def audio_array_to_signal(x, fs):
    """
    Konwersja z formatu audio na pf.Signal
    (n_samples, n_channels) -> (n_channels, n_samples)
    """
    x = np.array(x, dtype=np.float32)
    return pf.Signal(np.transpose(x), fs)

def play_and_record(signal, in_channels=1, device= None, blocking=True, latency="low"):
    """
    Odtwarzanie sygnału  i nagrywanie wejścia audio
    signal: sygnał testowy do odtworzenia (ESS/MLS)
    in_channels: liczba kanałów wejściowych nagrania
    device: id urządzenia albo (out_id, in_id) z sounddevice
    blocking: jeśli True, czeka do końca odtwarzania
    latency: opóźnienie, "low"/"high"/numeryczna
    """
    fs = int(signal.sampling_rate)
    x = signal_to_audio_array(signal)

    y = sd.playrec(x, samplerate=fs, channels=in_channels, device=device, blocking=blocking, latency=latency, dtype='float32')

    return audio_array_to_signal(y, fs)

def save_wav(path, signal):
    """
    Zapis plików do formatu WAV
    path: ścieżka plików
    signal: sygnał do zapisu
    """
    fs = int(signal.sampling_rate)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = signal_to_audio_array(signal)  # (n_samples, n_channels)
    sf.write(str(path), x, fs, subtype="FLOAT")

def main():
    # Konfiguracja parametrów
    fs = 48000
    duration = 10.0
    f1 = 20.0
    f2 = 20000.0
    in_channels = 2
    device_in = 1
    device_out = 3
    outdir = "ir_session"
    tag = "test"

    if device_out is not None and device_in is not None:
        device = (device_in, device_out)
    else:
        device = None

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generacja sygnału testowego
    ess = generate_ess(f1=f1, f2=f2, duration=duration, fs=fs)
    #mls = generate_mls(order=10, fs=fs)

    print("===Odtwarzanie sygnału testowego i nagrywanie===")
    recording = play_and_record(ess, in_channels=in_channels, device=device, blocking=True)

    # Dekonwolucja - obliczenie odpowiedzi impulsowej
    H = pf.dsp.deconvolve(system_output=recording, system_input=ess)
    ir_time = pf.Signal(H.time, fs)

    sweep_path = outdir / f"sygnal_testowy_{ts}_{tag}.wav"
    rec_path   = outdir / f"nagranie_{ts}_{tag}.wav"
    ir_path    = outdir / f"IR_{ts}_{tag}.wav"

    save_wav(sweep_path, ess)
    save_wav(rec_path, recording)
    save_wav(ir_path, ir_time)

    print("=== Zapisane pliki ===")
    print(f"Sygnal testowy: {sweep_path}")
    print(f"Nagranie: {rec_path}")
    print(f"IR: {ir_path}")

if __name__ == "__main__":
    main()
