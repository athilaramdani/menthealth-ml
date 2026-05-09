"""
G_fitur_lanjutan.py — Eksplorasi G: Fitur Tambahan dari PPT & Literatur
Mencakup item G1-G5 dari listexplorasi.md:
  G1 — MFCC (13-40 koefisien) langsung dari .wav
  G2 — Spectral Centroid
  G3 — Zero Crossing Rate (ZCR)
  G4 — Spectral Rolloff
  G5 — Jitter & Shimmer (dari COVAREP)
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import random
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorasi.config import (DATASET_DIR, OUTPUT_DIR, FILE_WAV, FILE_COVAREP,
                               COVAREP_COLS, SAMPLE_N, RANDOM_SEED)

OUT = os.path.join(OUTPUT_DIR, "G_fitur_lanjutan")
os.makedirs(OUT, exist_ok=True)
random.seed(RANDOM_SEED)

SR_TARGET  = 16000
N_MFCC     = 40
LOAD_DUR   = 120  # detik audio yang dimuat per file (untuk efisiensi)


def get_participant_ids():
    pids = []
    for name in sorted(os.listdir(DATASET_DIR)):
        if os.path.isdir(os.path.join(DATASET_DIR, name)) and name.endswith("_P"):
            pids.append(name.replace("_P", ""))
    return pids


def load_audio(pid):
    path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_WAV.format(pid=pid))
    if not os.path.isfile(path):
        return None, None
    try:
        y, sr = librosa.load(path, sr=SR_TARGET, mono=True, duration=LOAD_DUR)
        return y, sr
    except Exception:
        return None, None


# ── G1: Ekstraksi & Plot MFCC ────────────────────────────────────────────────
def g1_mfcc(samples):
    print("\n" + "═"*60)
    print(f"G1 — Ekstraksi MFCC ({N_MFCC} koefisien) langsung dari .wav")
    print("═"*60)

    records = []
    fig, axes = plt.subplots(2, len(samples)//2 + len(samples)%2, figsize=(18, 8))
    axes = axes.flatten()

    for i, (pid, ax) in enumerate(zip(samples, axes)):
        y, sr = load_audio(pid)
        if y is None:
            ax.set_visible(False)
            continue

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        img  = librosa.display.specshow(mfcc, x_axis="time", sr=sr, ax=ax, cmap="plasma")
        ax.set_title(f"MFCC — Partisipan {pid}", fontsize=9)
        ax.set_xlabel("Waktu (s)")
        ax.set_ylabel("Koefisien MFCC")
        fig.colorbar(img, ax=ax)

        # Simpan mean MFCC per koefisien
        row = {"participant_id": pid}
        for k in range(N_MFCC):
            row[f"mfcc_{k}_mean"] = mfcc[k].mean()
            row[f"mfcc_{k}_std"]  = mfcc[k].std()
        records.append(row)
        print(f"  [{pid}]  MFCC shape={mfcc.shape}")

    for ax in axes[len(samples):]:
        ax.set_visible(False)

    plt.suptitle(f"G1 — MFCC ({N_MFCC} koefisien) dari Audio .wav\n(Pertama {LOAD_DUR} detik, SR={SR_TARGET}Hz)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "g1_mfcc_samples.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")

    if records:
        df_mfcc = pd.DataFrame(records)
        csv_path = os.path.join(OUT, "g1_mfcc_stats.csv")
        df_mfcc.to_csv(csv_path, index=False)
        print(f"  📄  Data MFCC disimpan → {csv_path}")


# ── G2, G3, G4: Spectral Features ────────────────────────────────────────────
def g2_g3_g4_spectral(samples):
    print("\n" + "═"*60)
    print("G2/G3/G4 — Spectral Centroid, ZCR, Spectral Rolloff")
    print("═"*60)

    records = []

    for pid in samples:
        y, sr = load_audio(pid)
        if y is None:
            continue

        # G2 — Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        # G3 — Zero Crossing Rate
        zcr      = librosa.feature.zero_crossing_rate(y)[0]
        # G4 — Spectral Rolloff
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

        records.append({
            "participant_id":       pid,
            "centroid_mean":        centroid.mean(),
            "centroid_std":         centroid.std(),
            "zcr_mean":             zcr.mean(),
            "zcr_std":              zcr.std(),
            "rolloff_mean":         rolloff.mean(),
            "rolloff_std":          rolloff.std(),
        })
        print(f"  [{pid}]  Centroid={centroid.mean():.1f}Hz  ZCR={zcr.mean():.4f}  Rolloff={rolloff.mean():.1f}Hz")

    df = pd.DataFrame(records)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cols_to_plot = [("centroid_mean", "Spectral Centroid Mean (Hz)", "#E91E63"),
                    ("zcr_mean",      "Zero Crossing Rate Mean",      "#009688"),
                    ("rolloff_mean",  "Spectral Rolloff Mean (Hz)",   "#FF5722")]

    for ax, (col, label, color) in zip(axes, cols_to_plot):
        ax.bar(df["participant_id"], df[col], color=color, edgecolor="white")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Participant ID")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("G2/G3/G4 — Spectral Features per Partisipan (Sampel)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "g2_g3_g4_spectral_features.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")

    csv_path = os.path.join(OUT, "g2_g3_g4_spectral_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"  📄  Data disimpan → {csv_path}")


# ── G5: Jitter & Shimmer dari COVAREP ────────────────────────────────────────
def g5_jitter_shimmer(samples):
    print("\n" + "═"*60)
    print("G5 — Jitter & Shimmer dari COVAREP")
    print("═"*60)
    print("  ℹ️  Jitter: variasi periodik F0. Shimmer: variasi amplitudo antar periode.")
    print("  ℹ️  Kolom Shimmer = kolom ke-73 (index 72) di COVAREP.")

    records = []

    for pid in samples:
        path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_COVAREP.format(pid=pid))
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, header=None)
        n  = df.shape[1]

        # F0 (col 0) untuk hitung Jitter secara proxy
        f0 = df.iloc[:, 0].values
        voiced_f0 = f0[f0 > 0]

        jitter_proxy = np.mean(np.abs(np.diff(voiced_f0))) if len(voiced_f0) > 1 else np.nan

        # Shimmer dari col 73 jika ada
        shimmer_mean = df.iloc[:, 72].replace(0, np.nan).dropna().mean() if n > 72 else np.nan

        records.append({
            "participant_id": pid,
            "jitter_proxy":   round(jitter_proxy, 4) if not np.isnan(jitter_proxy) else None,
            "shimmer_mean":   round(shimmer_mean, 6) if not np.isnan(shimmer_mean) else None,
        })
        print(f"  [{pid}]  Jitter(proxy)={jitter_proxy:.4f}  Shimmer={shimmer_mean:.6f}")

    df_result = pd.DataFrame(records)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label, color in zip(axes,
        ["jitter_proxy", "shimmer_mean"],
        ["Jitter (variasi F0 antar frame voiced)", "Shimmer Mean (dari COVAREP)"],
        ["#FF5722", "#3F51B5"]):
        data = df_result[col].dropna()
        ax.bar(df_result["participant_id"][:len(data)], data, color=color, edgecolor="white")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Participant ID")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("G5 — Jitter & Shimmer per Partisipan (Sampel)\nIndikator ketidakstabilan suara",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "g5_jitter_shimmer.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")

    csv_path = os.path.join(OUT, "g5_jitter_shimmer.csv")
    df_result.to_csv(csv_path, index=False)
    print(f"  📄  Data disimpan → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EDA - G: FITUR LANJUTAN (PPT & LITERATUR)")
    print("  Dataset : DAIC-WOZ")
    print("="*60)

    all_pids = get_participant_ids()
    samples  = random.sample(all_pids, min(SAMPLE_N, len(all_pids)))
    print(f"  Total partisipan : {len(all_pids)}")
    print(f"  Sampel analisis  : {samples}")
    print(f"  (Tiap file audio dimuat {LOAD_DUR} detik pertama untuk efisiensi)")

    g1_mfcc(samples)
    g2_g3_g4_spectral(samples)
    g5_jitter_shimmer(samples)

    print("\n" + "="*60)
    print("  [OK] Selesai. Output tersimpan di:")
    print(f"  {OUT}")
    print("="*60 + "\n")
