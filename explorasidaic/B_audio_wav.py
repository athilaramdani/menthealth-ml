"""
B_audio_wav.py — Eksplorasi B: Analisis File Audio (.wav)
Mencakup item B1-B6 dari listexplorasi.md
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorasi.config import (DATASET_DIR, OUTPUT_DIR, FILE_WAV, SAMPLE_N, RANDOM_SEED)

OUT = os.path.join(OUTPUT_DIR, "B_audio_wav")
os.makedirs(OUT, exist_ok=True)
random.seed(RANDOM_SEED)


def get_participant_ids():
    pids = []
    for name in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, name)
        if os.path.isdir(path) and name.endswith("_P"):
            pids.append(name.replace("_P", ""))
    return pids


# ── B1 & B2: Distribusi durasi dan sampling rate ──────────────────────────────
def b1_b2_durasi_sampling_rate(all_pids):
    print("\n" + "═"*60)
    print("B1 & B2 — Distribusi Durasi & Sampling Rate")
    print("═"*60)
    print("  🔄  Membaca metadata semua file WAV (hanya header, cepat)...")

    durations   = []
    sr_counts   = {}
    errors      = []

    for pid in all_pids:
        wav_path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_WAV.format(pid=pid))
        if not os.path.isfile(wav_path):
            continue
        try:
            info = sf.info(wav_path)
            dur  = info.frames / info.samplerate
            durations.append((pid, dur))
            sr_counts[info.samplerate] = sr_counts.get(info.samplerate, 0) + 1
        except Exception as e:
            errors.append((pid, str(e)))

    dur_vals = [d for _, d in durations]
    print(f"\n  Total file terbaca : {len(durations)}")
    print(f"  Durasi (detik)     : min={min(dur_vals):.1f}  max={max(dur_vals):.1f}  mean={np.mean(dur_vals):.1f}  std={np.std(dur_vals):.1f}")
    print(f"  Durasi (menit)     : min={min(dur_vals)/60:.1f}  max={max(dur_vals)/60:.1f}  mean={np.mean(dur_vals)/60:.1f}")
    print(f"  Sampling Rate      : {sr_counts}")

    if errors:
        print(f"  ⚠️  Error baca {len(errors)} file.")

    # Plot histogram durasi
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist([d/60 for d in dur_vals], bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Durasi (menit)", fontsize=12)
    ax.set_ylabel("Jumlah Partisipan", fontsize=12)
    ax.set_title("B1 — Distribusi Durasi Rekaman Wawancara (DAIC-WOZ)", fontsize=14, fontweight="bold")
    ax.axvline(np.mean(dur_vals)/60, color="red", linestyle="--", label=f"Mean = {np.mean(dur_vals)/60:.1f} menit")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT, "b1_distribusi_durasi.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")

    # Simpan ringkasan
    summary_path = os.path.join(OUT, "b1_b2_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total file     : {len(durations)}\n")
        f.write(f"Durasi min     : {min(dur_vals):.2f} detik ({min(dur_vals)/60:.2f} menit)\n")
        f.write(f"Durasi max     : {max(dur_vals):.2f} detik ({max(dur_vals)/60:.2f} menit)\n")
        f.write(f"Durasi mean    : {np.mean(dur_vals):.2f} detik\n")
        f.write(f"Durasi std     : {np.std(dur_vals):.2f} detik\n")
        f.write(f"Sampling Rate  : {sr_counts}\n")
        if errors:
            f.write(f"\nError:\n")
            for pid, err in errors:
                f.write(f"  {pid}: {err}\n")
    print(f"  📄  Ringkasan → {summary_path}")

    return durations


# ── B3: Waveform beberapa sampel ──────────────────────────────────────────────
def b3_plot_waveform(all_pids):
    print("\n" + "═"*60)
    print("B3 — Plot Waveform Sampel Partisipan")
    print("═"*60)

    samples = random.sample(all_pids, min(SAMPLE_N // 2, len(all_pids)))
    fig, axes = plt.subplots(len(samples), 1, figsize=(14, 3 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for ax, pid in zip(axes, samples):
        wav_path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_WAV.format(pid=pid))
        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True, duration=60)  # load 60 detik pertama
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#2196F3", alpha=0.8)
            ax.set_title(f"Partisipan {pid} (60 detik pertama, SR={sr}Hz)", fontsize=10)
            ax.set_ylabel("Amplitude")
        except Exception as e:
            ax.set_title(f"Partisipan {pid} — ERROR: {e}")

    plt.suptitle("B3 — Waveform Sampel Partisipan (DAIC-WOZ)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUT, "b3_waveform_samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Plot ({len(samples)} sampel) disimpan → {out_path}")


# ── B4 & B5: RMS Energy & Clipping ───────────────────────────────────────────
def b4_b5_rms_clipping(all_pids):
    print("\n" + "═"*60)
    print("B4 & B5 — RMS Energy & Deteksi Clipping")
    print("═"*60)

    samples = random.sample(all_pids, min(SAMPLE_N, len(all_pids)))
    rms_vals    = []
    clip_report = []

    for pid in samples:
        wav_path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_WAV.format(pid=pid))
        if not os.path.isfile(wav_path):
            continue
        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            rms   = float(np.sqrt(np.mean(y**2)))
            clip  = float(np.mean(np.abs(y) > 0.99))  # fraksi frame clipping
            rms_vals.append((pid, rms))
            if clip > 0.001:
                clip_report.append((pid, clip))
            print(f"    [{pid}]  RMS={rms:.5f}  Clipping={clip*100:.3f}%")
        except Exception as e:
            print(f"    [{pid}]  ERROR: {e}")

    if rms_vals:
        rms_only = [v for _, v in rms_vals]
        print(f"\n  RMS mean={np.mean(rms_only):.5f}  std={np.std(rms_only):.5f}")

        # Plot RMS
        fig, ax = plt.subplots(figsize=(10, 4))
        pids_s  = [p for p, _ in rms_vals]
        ax.bar(pids_s, rms_only, color="#4CAF50", edgecolor="white")
        ax.set_xlabel("Participant ID", fontsize=11)
        ax.set_ylabel("RMS Energy", fontsize=11)
        ax.set_title("B4 — RMS Energy per Partisipan (Sampel)", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        out_path = os.path.join(OUT, "b4_rms_energy.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  📊  Plot RMS disimpan → {out_path}")

    if clip_report:
        print(f"\n  ⚠️  Clipping terdeteksi pada {len(clip_report)} file:")
        for pid, pct in clip_report:
            print(f"      [{pid}]  {pct*100:.3f}%")
    else:
        print("  ✅  Tidak ada clipping signifikan terdeteksi.")


# ── B6: Mel-Spectrogram ────────────────────────────────────────────────────────
def b6_mel_spectrogram(all_pids):
    print("\n" + "═"*60)
    print("B6 — Plot Mel-Spectrogram Sampel")
    print("═"*60)

    samples = random.sample(all_pids, min(4, len(all_pids)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, pid in zip(axes, samples):
        wav_path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_WAV.format(pid=pid))
        try:
            y, sr = librosa.load(wav_path, sr=16000, mono=True, duration=60)
            S     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_db  = librosa.power_to_db(S, ref=np.max)
            img   = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                             fmax=8000, ax=ax, cmap="viridis")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title(f"Partisipan {pid}", fontsize=11)
        except Exception as e:
            ax.set_title(f"{pid} — ERROR")
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)

    plt.suptitle("B6 — Mel-Spectrogram Sampel Partisipan DAIC-WOZ\n(60 detik pertama, SR=16kHz)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "b6_mel_spectrogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EDA - B: EKSPLORASI FILE AUDIO (.wav)")
    print("  Dataset : DAIC-WOZ")
    print("="*60)

    all_pids = get_participant_ids()
    print(f"  Total partisipan tersedia: {len(all_pids)}")

    b1_b2_durasi_sampling_rate(all_pids)
    b3_plot_waveform(all_pids)
    b4_b5_rms_clipping(all_pids)
    b6_mel_spectrogram(all_pids)

    print("\n" + "="*60)
    print("  [OK] Selesai. Output tersimpan di:")
    print(f"  {OUT}")
    print("="*60 + "\n")
