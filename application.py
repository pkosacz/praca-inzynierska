import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
    number_ir_channels = ir_signal.cshape[0]
    total_plots = 2 + number_ir_channels

    fig, ax = plt.subplots(total_plots, 1, figsize=(10, 3 * total_plots), constrained_layout=True)

    # 1. Sygnał testowy (input)
    pf.plot.time(test_signal, ax=ax[0], dB=False, unit='s')
    ax[0].set_title("1. Sygnał testowy (input)")
    ax[0].set_ylabel("Amplituda")

    # 2. Sygnał nagrany (output)
    pf.plot.time(rec_signal, ax=ax[1], dB=False, unit='s')
    ax[1].set_title("2. Sygnał Nagrany (output)")
    ax[1].set_ylabel("Amplituda")

    # 3. Odpowiedź Impulsowa
    for i in range(number_ir_channels):
        plot_index = 2 + i
        current_ir_channel = ir_signal[i]
        pf.plot.time(current_ir_channel, ax=ax[plot_index], dB=True, unit='s')
        ax[plot_index].set_title(f"3.{i+1} Obliczona Odpowiedź Impulsowa - Kanał {i+1}")
        ax[plot_index].set_ylabel("Amplituda (dB)")

    plt.show()

class ImpulseResponseApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacja do Pomiaru IR")
        self.root.geometry("500x650")

        self.output_dir = Path("ir_session").resolve()

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

        ttk.Label(root, text='Wybierz Sygnał Testowy:').pack(pady=(10, 2))
        self.signal_var = tk.StringVar(value="ESS")
        self.signal_combo = ttk.Combobox(root, width=20, textvariable=self.signal_var, state="readonly")
        self.signal_combo["values"] = ("ESS", "MLS")
        self.signal_combo.pack(pady=5)
        self.signal_combo.bind("<<ComboboxSelected>>", self.toggle_mls_order_field)

        self.params_container = ttk.Frame(root)
        self.params_container.pack(pady=5)

        self.frame_duration = ttk.Frame(self.params_container)
        ttk.Label(self.frame_duration, text="Czas trwania ESS (s):").pack(side="left", padx=5)
        self.duration_entry = ttk.Entry(self.frame_duration, width=10)
        self.duration_entry.insert(0, "5.0")
        self.duration_entry.pack(side="left", padx=5)

        self.frame_mls_order = ttk.Frame(self.params_container)
        self.mls_label = ttk.Label(self.frame_mls_order, text="Rząd MLS (np. 10):")
        self.mls_label.pack(side="left", padx=5)
        self.mls_order_entry = ttk.Entry(self.frame_mls_order, width=10)
        self.mls_order_entry.insert(0, "10")
        self.mls_order_entry.pack(side="left", padx=5)

        self.toggle_mls_order_field()

        self.position_frame = ttk.Frame(root)
        self.position_frame.pack(pady=5)

        ttk.Label(self.position_frame, text="Source").pack(side="left", padx=(0, 5))
        self.source_id = tk.IntVar(value=1)
        self.source_spin = ttk.Spinbox(self.position_frame, from_=1, to=10, textvariable=self.source_id, width=4, state="readonly")
        self.source_spin.pack(side="left", padx=(0, 20))

        ttk.Label(self.position_frame, text="Receiver").pack(side="left", padx=(0, 5))
        self.receiver_id = tk.IntVar(value=1)
        self.receiver_spin = ttk.Spinbox(self.position_frame, from_=1, to=10, textvariable=self.receiver_id, width=4, state="readonly")
        self.receiver_spin.pack(side="left")

        self.folder_frame = ttk.LabelFrame(root, text="Lokaliacja Zapisu")
        self.folder_frame.pack(pady=10, padx=10, fill="x")

        self.button_browse = ttk.Button(self.folder_frame, text="Wybierz folder", command=self.choose_directory)
        self.button_browse.pack(side="left", pady=5, padx=5)

        self.path_label = ttk.Label(self.folder_frame, text=str(self.output_dir), wraplength=350)
        self.path_label.pack(side="left", pady=5, padx=5)

        self.start_btn = ttk.Button(root, text="ROZPOCZNIJ POMIAR", command=self.start_measurement_thread)
        self.start_btn.pack(pady=20, ipady=5)

        self.log_text = tk.Text(root, height=8, width=55, state='disabled')
        self.log_text.pack(pady=5)

        self.log("Gotowy do pracy. Wybierz urządzenia i sygnał testowy.")

    def choose_directory(self):
        selected_dir = filedialog.askdirectory(initialdir=self.output_dir)
        if selected_dir:
            self.output_dir = Path(selected_dir)
            self.path_label.config(text=str(self.output_dir))
            self.log(f"Zmieniono ścieżkę zapisu na {self.output_dir.name}")

    def toggle_mls_order_field(self, event=None):
        if self.signal_var.get() == "MLS":
            self.frame_duration.pack_forget()
            self.frame_mls_order.pack(pady=5)
        else:
            self.frame_mls_order.pack_forget()
            self.frame_duration.pack(pady=5)

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
            signal_type = self.signal_var.get()

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
            self.log(f"In: {id_in}, Out: {id_out}")

            test_signal = None
            duration_ir_padding = 1.0

            if signal_type == "ESS":
                try:
                    duration = float(self.duration_entry.get())
                except ValueError:
                    duration = 10.0

                self.log(f"Generowanie ESS (Czas: {duration}s)")
                ess = generate_ess(f1, f2, duration, fs)
                test_signal = pf.dsp.pad_zeros(ess, pad_width=int(duration_ir_padding * fs))

            elif signal_type == "MLS":
                try:
                    order = int(self.mls_order_entry.get())
                except ValueError:
                    order = 10

                self.log(f"Generowanie MLS (Rząd: {order}")
                mls = generate_mls(order, fs)
                test_signal = pf.dsp.pad_zeros(mls, pad_width=int(duration_ir_padding * fs))

            if test_signal is None:
                messagebox.showerror("Błąd", "Nie udało się wygenerować sygnału testowego")
                return

            self.log("Nagrywanie w toku")
            recording = play_and_record(
                test_signal,
                in_channels=in_channels,
                device=(id_in, id_out),
                blocking=True
            )
            self.log("Nagrywanie zakończone")

            self.log("Obliczanie odpowiedzi impulsowej")

            ir_a_format = pf.dsp.deconvolve(
                system_output=recording,
                system_input=test_signal,
                frequency_range=(f1, f2) if signal_type == "ESS" else None,
            )

            try:
                start_sample = pf.dsp.find_impulse_response_start(ir_a_format, threshold=20)
                ir_aligned = pf.dsp.time_shift(ir_a_format, -start_sample[0])
            except:
                ir_aligned = ir_a_format

            ir_final = convert_a_to_b_format(ir_aligned)

            source_value = self.source_id.get()
            receiver_value = self.receiver_id.get()

            outdir = self.output_dir
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

            file_prefix = f"S{source_value}_R{receiver_value}_{ts}"

            outdir.mkdir(parents=True, exist_ok=True)

            sweep_path = outdir / f"sygnal_testowy_{file_prefix}.wav"
            rec_path = outdir / f"nagranie_{file_prefix}.wav"
            ir_path = outdir / f"IR_{file_prefix}.wav"

            save_wav(sweep_path, test_signal)
            save_wav(rec_path, recording)
            pfio.write_audio(ir_final, str(ir_path), "FLOAT")

            self.log("Zapisano pliki")
            self.log(f"Sygnal testowy: {sweep_path}")
            self.log(f"Nagranie: {rec_path}")
            self.log(f"IR: {ir_path}")

            self.root.after(0, self.finish_measurement, test_signal, recording, ir_final)

        except Exception as e:
            self.log(f"BŁĄD: {e}")
            messagebox.showerror("Błąd krytyczny", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImpulseResponseApplication(root)
    root.mainloop()