import tkinter as tk
from tkinter import ttk, messagebox
import threading
from pathlib import Path
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import sounddevice as sd
import pyfar as pf
from pyfar import io as pfio
from scipy.signal import max_len_seq

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


class ImpulseResponseApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacja do Pomiaru IR")
        self.root.geometry("500x450")

        self.device_list = sd.query_devices()
        self.host_apis = sd.query_hostapis()

        self.input_device_ids = []
        self.output_device_ids = []

        ttk.Label(root, text="Wybierz Urządzenie Wejściowe (Input):").pack(pady=(10, 2))
        self.input_combo = ttk.Combobox(root, width=70, state="readonly")
        self.input_combo.pack(pady=5)

        ttk.Label(root, text="Wybierz Urządzenie Wyjściowe (Output):").pack(pady=(10, 2))
        self.output_combo = ttk.Combobox(root, width=70, state="readonly")
        self.output_combo.pack(pady=5)

        self.get_device_lists()

        frame_params = ttk.Frame(root)
        frame_params.pack(pady=10)

        ttk.Label(frame_params, text="Czas trwania (s):").pack(side="left", padx=5)

        self.duration_entry = ttk.Entry(frame_params, width=10)
        self.duration_entry.insert(0, "5.0")
        self.duration_entry.pack(side="left", padx=5)

        self.start_btn = ttk.Button(root, text="ROZPOCZNIJ POMIAR", command=self.start_measurement_thread)
        self.start_btn.pack(pady=20, ipady=5)

        self.log_text = tk.Text(root, height=8, width=55, state='disabled')
        self.log_text.pack(pady=5)

        self.log("Gotowy do pracy. Wybierz urządzenia.")

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def get_device_lists(self):
        input_descriptions = []
        output_descriptions = []

        self.input_device_ids = []
        self.output_device_ids = []

        for index, device in enumerate(self.device_list):
            name = device['name']
            api = self.host_apis[device['hostapi']]['name']
            in_channels = device['max_input_channels']
            out_channels = device['max_output_channels']
            description = f"[{index}] {name}, {api} ({in_channels} in, {out_channels} out)"

            if in_channels > 0:
                input_descriptions.append(description)
                self.input_device_ids.append(index)

            if out_channels > 0:
                output_descriptions.append(description)
                self.output_device_ids.append(index)

        self.input_combo['values'] = input_descriptions
        self.output_combo['values'] = output_descriptions

        if input_descriptions: self.input_combo.current(0)
        if output_descriptions: self.output_combo.current(0)

    def start_measurement_thread(self):
        thread = threading.Thread(target=self.run_measurement_logic)
        thread.start()

    def finish_measurement(self, test_signal, recording, ir_final):
        plot_results(test_signal, recording, ir_final)

        self.log("--- KONIEC ---")
        self.start_btn.config(state='normal')

    def run_measurement_logic(self):
        self.start_btn.config(state='disabled')

        try:
            input_index = self.input_combo.current()
            output_index = self.output_combo.current()

            if input_index == -1 or output_index == -1:
                messagebox.showerror("Błąd", "Nie wybrano urządzeń!")
                return

            id_in = self.input_device_ids[input_index]
            id_out = self.output_device_ids[output_index]

            try:
                duration = float(self.duration_entry.get())
            except ValueError:
                duration = 5.0

            fs = 48000
            f1 = 20.0
            f2 = 20000.0
            in_channels = 2

            self.log(f"--- START POMIARU ---")
            self.log(f"In: {id_in}, Out: {id_out}, Czas: {duration}s")

            self.log("Generowanie sygnału testowego")
            ess = generate_ess(f1, f2, duration, fs)
            ess_padded = pf.dsp.pad_zeros(ess, pad_width=int(1.0 * fs))

            self.log("Nagrywanie w toku")
            recording = play_and_record(
                ess_padded,
                in_channels=in_channels,
                device=(id_in, id_out),
                blocking=True
            )
            self.log("Nagrywanie zakończone")

            self.log("Obliczanie odpowiedzi impulsowej")
            ir_a_format = pf.dsp.deconvolve(
                system_output=recording,
                system_input=ess_padded,
                frequency_range=(f1, f2),
            )

            try:
                start_sample = pf.dsp.find_impulse_response_start(ir_a_format, threshold=20)
                ir_aligned = pf.dsp.time_shift(ir_a_format, -start_sample[0])
            except:
                ir_aligned = ir_a_format

            ir_final = convert_a_to_b_format(ir_aligned)

            outdir = Path("ir_session")
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir.mkdir(parents=True, exist_ok=True)

            sweep_path = outdir / f"sygnal_testowy_{ts}.wav"
            rec_path = outdir / f"nagranie_{ts}.wav"
            ir_path = outdir / f"IR_{ts}.wav"

            save_wav(sweep_path, ess_padded)
            save_wav(rec_path, recording)
            pfio.write_audio(ir_final, str(ir_path), "FLOAT")

            self.log("Zapisano pliki")
            self.log(f"Sygnal testowy: {sweep_path}")
            self.log(f"Nagranie: {rec_path}")
            self.log(f"IR: {ir_path}")

            self.root.after(0, self.finish_measurement, ess_padded, recording, ir_final)

        except Exception as e:
            self.log(f"BŁĄD: {e}")
            messagebox.showerror("Błąd krytyczny", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImpulseResponseApplication(root)
    root.mainloop()