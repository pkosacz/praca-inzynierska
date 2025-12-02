from pathlib import Path

import matplotlib.pyplot as plt
from scipy.signal import max_len_seq
from pyfar import io as pfio

import numpy as np
import soundfile as sf
import sounddevice as sd
import pyfar as pf
import datetime as dt

def generate_ess(f1=20.0, f2=20000.0, duration=10.0, fs=48000):
    """
    Generacja sygnału Exponential Sine Sweep (ESS) w dziedzinie czasu
    f1-f2: zakres częstotliwości
    duration: czas trwania sygnału
    fs: częstotliwość próbkowania
    """
    n_samples = int(fs * duration)

    ess_signal = pf.signals.exponential_sweep_time(
        n_samples=n_samples,
        frequency_range=[f1, f2],
        sampling_rate=fs,
    )

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

def convert_a_to_b_format(signal_a_format):
    conversion_matrix = np.array([
        #[1.0, 1.0, 1.0, 1.0],   # W
        #[1.0, 1.0, -1.0, -1.0], # X
        #[1.0, -1.0, 1.0, -1.0], # Y
        #[1.0, -1.0, -1.0, 1.0]  # Z
        [1.0, 1.0],
        [1.0, -1.0]
    ])

    data = signal_a_format.time
    b_format_data = conversion_matrix @ data

    return pf.Signal(b_format_data, signal_a_format.sampling_rate)

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

    y = sd.playrec(
        data=x,
        samplerate=fs,
        channels=in_channels,
        device=device,
        blocking=blocking,
        latency=latency,
        dtype='float32',
    )

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


def plot_results(test_signal, rec_signal, ir_signal):
    """
    Rysowanie trzech wykresów:
    1. Sygnał testowy
    2. Sygnał nagrany
    3. Obliczona odpowiedź impulsowa (IR)
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    # 1. Sygnał testowy (input)
    pf.plot.time(test_signal, ax=ax[0], dB=False, unit='s')
    ax[0].set_title("1. Sygnał testowy (input)")
    ax[0].set_ylabel("Amplituda")

    # 2. Sygnał nagrany (output)
    pf.plot.time(rec_signal, ax=ax[1], dB=False, unit='s')
    ax[1].set_title("2. Sygnał Nagrany (output)")
    ax[1].set_ylabel("Amplituda")

    # 3. Odpowiedź Impulsowa
    pf.plot.time(ir_signal, ax=ax[2], dB=True, unit='s')
    ax[2].set_title("3. Obliczona Odpowiedź Impulsowa")
    ax[2].set_ylabel("Amplituda (dB)")

    plt.tight_layout()

    plt.show()


def main():
    # Konfiguracja parametrów
    fs = 48000
    duration = 10.0
    f1 = 20.0
    f2 = 20000.0

    in_channels = 2
    device_in = 1
    device_out = 3

    outdir = Path("ir_session")
    tag = "test"

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    # Generacja sygnału testowego
    ess = generate_ess(f1=f1, f2=f2, duration=duration, fs=fs)
    # Dodanie paddingu
    ess_padded = pf.dsp.pad_zeros(ess, pad_width=int(1 * fs))
    #mls = generate_mls(order=10, fs=fs)

    print("===Odtwarzanie sygnału testowego i nagrywanie===")
    recording = play_and_record(
        ess_padded,
        in_channels=in_channels,
        device=(device_in, device_out),
        blocking=True,
        latency="low",
    )

    # Dekonwolucja - obliczenie odpowiedzi impulsowej
    ir_a_format = pf.dsp.deconvolve(
        system_output=recording,
        system_input=ess_padded,
        frequency_range=(f1, f2),
    )

    # Znalezienie początku odpowiedzi impulsowej i przesunięcie wykresu do tego punktu
    try:
        start_sample = pf.dsp.find_impulse_response_start(ir_a_format, threshold=20)
        ir_aligned = pf.dsp.time_shift(ir_a_format, -start_sample[0])
    except Exception as e:
        print(f"Nie udało się automatycznie wykryć początku impulsu: {e}")
        ir_aligned = ir_a_format

    ir_final = convert_a_to_b_format(ir_aligned)

    sweep_path = outdir / f"sygnal_testowy_{ts}_{tag}.wav"
    rec_path   = outdir / f"nagranie_{ts}_{tag}.wav"
    ir_path    = outdir / f"IR_{ts}_{tag}.wav"

    save_wav(sweep_path, ess_padded)
    save_wav(rec_path, recording)

    pfio.write_audio(ir_final, str(ir_path), "FLOAT")

    print("=== Zapisane pliki ===")
    print(f"Sygnal testowy: {sweep_path}")
    print(f"Nagranie: {rec_path}")
    print(f"IR: {ir_path}")

    plot_results(ess_padded, recording, ir_final)

if __name__ == "__main__":
    main()
