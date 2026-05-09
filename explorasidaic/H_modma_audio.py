"""
H_modma_audio.py — Eksplorasi Khusus Dataset Audio MODMA
Menganalisis distribusi jumlah file, durasi, sampling rate, dan plot waveform.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tambahkan base dir ke path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Konfigurasi path khusus MODMA
MODMA_DIR = os.path.join(BASE_DIR, "dataset", "raw", "MODMA")
EXCEL_PATH = os.path.join(MODMA_DIR, "subjects_information_audio_lanzhou_2015.xlsx")
OUT = os.path.join(BASE_DIR, "explorasi", "output", "H_modma_audio")

os.makedirs(OUT, exist_ok=True)
random.seed(42)

def load_modma_metadata():
    print(f"  🔄  Membaca metadata dari {EXCEL_PATH}...")
    try:
        df = pd.read_excel(EXCEL_PATH)
        # Ambil kolom yang relevan: subject id dan type (MDD/HC)
        df_meta = df[['subject id', 'type']].dropna()
        # Ubah subject id jadi string dengan padding nol di depan (8 karakter, karena folder namanya misal 02010001)
        # Tapi id di excel itu 2010001 (7 digit). Kita pad dengan 0 jadi 8 digit.
        df_meta['folder_name'] = df_meta['subject id'].apply(lambda x: str(int(x)).zfill(8))
        metadata_dict = pd.Series(df_meta['type'].values, index=df_meta['folder_name']).to_dict()
        print(f"  ✅  Berhasil membaca {len(metadata_dict)} data partisipan.")
        
        # Cetak sebaran label
        counts = df_meta['type'].value_counts()
        print(f"  📊  Sebaran Kelas: {counts.to_dict()}")
        return metadata_dict
    except Exception as e:
        print(f"  ❌  Gagal membaca metadata: {e}")
        return {}

def h1_h2_analisis_durasi_dan_jumlah(metadata):
    print("\n" + "═"*60)
    print("H1 & H2 — Analisis Jumlah File, Durasi & Sampling Rate")
    print("═"*60)
    
    stats = []
    sr_counts = {}
    
    for folder_name in os.listdir(MODMA_DIR):
        folder_path = os.path.join(MODMA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Cari labelnya (MDD/HC)
        label = metadata.get(folder_name, "UNKNOWN")
        
        # Hitung jumlah file wav dan durasi
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        total_duration = 0.0
        file_durations = []
        
        for wf in wav_files:
            wf_path = os.path.join(folder_path, wf)
            try:
                info = sf.info(wf_path)
                dur = info.frames / info.samplerate
                total_duration += dur
                file_durations.append(dur)
                sr_counts[info.samplerate] = sr_counts.get(info.samplerate, 0) + 1
            except Exception as e:
                pass
                
        if len(wav_files) > 0:
            stats.append({
                'pid': folder_name,
                'label': label,
                'num_files': len(wav_files),
                'total_duration': total_duration,
                'avg_duration': np.mean(file_durations)
            })
            
    df_stats = pd.DataFrame(stats)
    
    # Ringkasan di terminal
    print("\n  [Ringkasan Keseluruhan]")
    print(f"  Total Partisipan dgn Audio : {len(df_stats)}")
    print(f"  Total File .wav terbaca    : {df_stats['num_files'].sum()}")
    print(f"  Total Durasi (jam)         : {df_stats['total_duration'].sum() / 3600:.2f} jam")
    print(f"  Sampling Rate              : {sr_counts}")
    
    print("\n  [Rata-rata per Partisipan berdasarkan Kelas]")
    avg_stats = df_stats.groupby('label')[['num_files', 'total_duration', 'avg_duration']].mean()
    print(avg_stats.round(2))
    
    # Plot 1: Boxplot Jumlah File per Partisipan (MDD vs HC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    mdd_num = df_stats[df_stats['label'] == 'MDD']['num_files']
    hc_num = df_stats[df_stats['label'] == 'HC']['num_files']
    axes[0].boxplot([mdd_num, hc_num], labels=['MDD', 'HC'])
    axes[0].set_title("H1 — Distribusi Jumlah File .wav per Partisipan")
    axes[0].set_ylabel("Jumlah File")
    
    # Plot 2: Boxplot Total Durasi per Partisipan
    mdd_dur = df_stats[df_stats['label'] == 'MDD']['total_duration']
    hc_dur = df_stats[df_stats['label'] == 'HC']['total_duration']
    axes[1].boxplot([mdd_dur, hc_dur], labels=['MDD', 'HC'])
    axes[1].set_title("H1 — Distribusi Total Durasi Audio per Partisipan")
    axes[1].set_ylabel("Total Durasi (detik)")
    
    plt.tight_layout()
    plot_path1 = os.path.join(OUT, "h1_distribusi_jumlah_dan_durasi.png")
    plt.savefig(plot_path1, dpi=150)
    plt.close()
    print(f"\n  📊  Plot disimpan → {plot_path1}")
    
    # Simpan Ringkasan ke TXT
    txt_path = os.path.join(OUT, "h1_h2_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=== Ringkasan Durasi & Jumlah File MODMA ===\n\n")
        f.write(f"Total Partisipan: {len(df_stats)}\n")
        f.write(f"Sampling Rates: {sr_counts}\n\n")
        f.write("Rata-rata per Partisipan berdasarkan Kelas:\n")
        f.write(avg_stats.to_string())
    print(f"  📄  Ringkasan disimpan → {txt_path}")
    
    return df_stats

def h3_visualisasi_sampel(df_stats):
    print("\n" + "═"*60)
    print("H3 — Visualisasi Waveform & Spectrogram (Sampel)")
    print("═"*60)
    
    # Ambil 1 partisipan MDD dan 1 HC
    mdd_pids = df_stats[df_stats['label'] == 'MDD']['pid'].tolist()
    hc_pids = df_stats[df_stats['label'] == 'HC']['pid'].tolist()
    
    if not mdd_pids or not hc_pids:
        print("  ❌  Tidak cukup data untuk divisualisasikan.")
        return
        
    sample_pids = {
        'MDD': random.choice(mdd_pids),
        'HC': random.choice(hc_pids)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for i, (label, pid) in enumerate(sample_pids.items()):
        folder_path = os.path.join(MODMA_DIR, pid)
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        if not wav_files:
            continue
            
        # Ambil file wav acak dari partisipan tersebut
        sample_wav = random.choice(wav_files)
        wav_path = os.path.join(folder_path, sample_wav)
        
        try:
            # Ambil 10 detik pertama biar gak kelamaan ngeplot
            y, sr = librosa.load(wav_path, sr=None, duration=10)
            
            # Plot Waveform
            ax_wf = axes[i, 0]
            librosa.display.waveshow(y, sr=sr, ax=ax_wf, color="#FF5722" if label == "MDD" else "#2196F3")
            ax_wf.set_title(f"Waveform - {label} (PID: {pid}, File: {sample_wav})")
            ax_wf.set_ylabel("Amplitude")
            
            # Plot Mel-Spectrogram
            ax_mel = axes[i, 1]
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000, ax=ax_mel, cmap="viridis")
            fig.colorbar(img, ax=ax_mel, format="%+2.0f dB")
            ax_mel.set_title(f"Mel-Spectrogram - {label} (10 detik pertama)")
            
        except Exception as e:
            print(f"  ❌  Error plotting {label} {pid}: {e}")
            
    plt.tight_layout()
    plot_path = os.path.join(OUT, "h3_waveform_spectrogram_samples.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  📊  Plot sampel disimpan → {plot_path}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EDA - H: EKSPLORASI FILE AUDIO (.wav) MODMA")
    print("="*60)
    
    metadata = load_modma_metadata()
    if metadata:
        df_stats = h1_h2_analisis_durasi_dan_jumlah(metadata)
        h3_visualisasi_sampel(df_stats)
    
    print("\n" + "="*60)
    print("  [OK] Selesai. Output tersimpan di:")
    print(f"  {OUT}")
    print("="*60 + "\n")
