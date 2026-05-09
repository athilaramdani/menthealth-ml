"""
C_covarep.py — Eksplorasi C: Analisis Fitur Akustik COVAREP.csv
Mencakup item C1-C7 dari listexplorasi.md
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorasi.config import (DATASET_DIR, OUTPUT_DIR, FILE_COVAREP,
                               COVAREP_COLS, SAMPLE_N, RANDOM_SEED)

OUT = os.path.join(OUTPUT_DIR, "C_covarep")
os.makedirs(OUT, exist_ok=True)
random.seed(RANDOM_SEED)

# Index kolom penting (0-indexed, sesuai COVAREP docs)
IDX_F0     = 0   # Fundamental frequency / pitch
IDX_VUV    = 1   # Voiced/Unvoiced
IDX_NAQ    = 2   # Normalized Amplitude Quotient (glottal)
IDX_QOQ    = 3   # Quasi-Open Quotient (glottal)
IDX_H1H2   = 4   # H1-H2 (glottal)
IDX_MCEP0  = 10  # MCEP coefficient 0 (energy-related)


def get_participant_ids():
    pids = []
    for name in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, name)
        if os.path.isdir(path) and name.endswith("_P"):
            pids.append(name.replace("_P", ""))
    return pids


def load_covarep(pid):
    """Load COVAREP.csv untuk satu partisipan, kembalikan DataFrame."""
    path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_COVAREP.format(pid=pid))
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, header=None)
    # Sesuaikan jumlah kolom dengan header yang tersedia
    n_cols = df.shape[1]
    cols   = COVAREP_COLS[:n_cols] if n_cols <= len(COVAREP_COLS) else \
             COVAREP_COLS + [f"extra_{i}" for i in range(n_cols - len(COVAREP_COLS))]
    df.columns = cols
    return df


# ── C1: Dimensi file COVAREP ──────────────────────────────────────────────────
def c1_dimensi(samples):
    print("\n" + "═"*60)
    print("C1 — Dimensi COVAREP.csv")
    print("═"*60)

    rows = []
    for pid in samples:
        df = load_covarep(pid)
        if df is not None:
            rows.append({"participant_id": pid, "n_frames": len(df), "n_cols": df.shape[1]})
            print(f"  [{pid}]  frames={len(df):6d}  cols={df.shape[1]}")

    summary = pd.DataFrame(rows)
    print(f"\n  Rata-rata frame : {summary['n_frames'].mean():.0f}")
    print(f"  Min frame       : {summary['n_frames'].min()}")
    print(f"  Max frame       : {summary['n_frames'].max()}")
    print(f"  Kolom           : {summary['n_cols'].iloc[0]} (konsisten={summary['n_cols'].nunique()==1})")

    out_path = os.path.join(OUT, "c1_dimensi.csv")
    summary.to_csv(out_path, index=False)
    print(f"  📄  Tabel disimpan → {out_path}")


# ── C2: Persentase silent frames ──────────────────────────────────────────────
def c2_silent_frames(all_pids):
    print("\n" + "═"*60)
    print("C2 — Persentase Silent Frames (F0 = 0)")
    print("═"*60)

    records = []
    print("  🔄  Memproses semua partisipan...")
    for pid in all_pids:
        df = load_covarep(pid)
        if df is None:
            continue
        f0 = df.iloc[:, IDX_F0]
        total   = len(f0)
        silent  = (f0 == 0).sum()
        records.append({"participant_id": pid, "total_frames": total,
                         "silent_frames": silent, "silent_pct": silent / total * 100})

    df_result = pd.DataFrame(records)
    print(f"\n  Rata-rata silent : {df_result['silent_pct'].mean():.1f}%")
    print(f"  Min silent       : {df_result['silent_pct'].min():.1f}%")
    print(f"  Max silent       : {df_result['silent_pct'].max():.1f}%")

    # Plot distribusi
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_result["silent_pct"], bins=30, color="#E74C3C", edgecolor="white")
    ax.axvline(df_result["silent_pct"].mean(), color="black", linestyle="--",
               label=f"Mean = {df_result['silent_pct'].mean():.1f}%")
    ax.set_xlabel("Persentase Silent Frames (%)", fontsize=12)
    ax.set_ylabel("Jumlah Partisipan", fontsize=12)
    ax.set_title("C2 — Distribusi Silent Frames per Partisipan\n(F0=0 artinya unvoiced/diam)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT, "c2_silent_frames.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")

    csv_path = os.path.join(OUT, "c2_silent_frames.csv")
    df_result.to_csv(csv_path, index=False)
    print(f"  📄  Data disimpan → {csv_path}")

    return df_result


# ── C3: Distribusi F0 (Pitch) ────────────────────────────────────────────────
def c3_pitch_distribution(samples):
    print("\n" + "═"*60)
    print("C3 — Distribusi Pitch / F0 (Voiced Frames Only)")
    print("═"*60)

    fig, axes = plt.subplots(2, len(samples)//2 + len(samples)%2, figsize=(16, 8))
    axes = axes.flatten()

    for i, (pid, ax) in enumerate(zip(samples, axes)):
        df = load_covarep(pid)
        if df is None:
            ax.set_visible(False)
            continue
        f0_voiced = df.iloc[:, IDX_F0][df.iloc[:, IDX_F0] > 0]
        ax.hist(f0_voiced, bins=40, color="#9B59B6", edgecolor="white", alpha=0.8)
        ax.set_title(f"{pid}\nMedian F0={f0_voiced.median():.1f} Hz", fontsize=9)
        ax.set_xlabel("F0 (Hz)")

    for ax in axes[len(samples):]:
        ax.set_visible(False)

    plt.suptitle("C3 — Distribusi Pitch (F0) per Partisipan\n(hanya voiced frames, F0 > 0)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "c3_pitch_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")


# ── C4: Distribusi fitur glottal (NAQ, QOQ, H1H2) ────────────────────────────
def c4_glottal_features(samples):
    print("\n" + "═"*60)
    print("C4 — Distribusi Fitur Glottal (NAQ, QOQ, H1-H2)")
    print("═"*60)

    all_naq, all_qoq, all_h1h2 = [], [], []

    for pid in samples:
        df = load_covarep(pid)
        if df is None:
            continue
        voiced = df[df.iloc[:, IDX_VUV] == 1]
        if len(voiced) == 0:
            continue
        all_naq.extend(voiced.iloc[:, IDX_NAQ].dropna().tolist())
        all_qoq.extend(voiced.iloc[:, IDX_QOQ].dropna().tolist())
        all_h1h2.extend(voiced.iloc[:, IDX_H1H2].dropna().tolist())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, name, color in zip(
        axes,
        [all_naq, all_qoq, all_h1h2],
        ["NAQ", "QOQ", "H1-H2"],
        ["#E91E63", "#FF9800", "#3F51B5"]
    ):
        if data:
            arr = np.array(data)
            # Clip outlier untuk visualisasi
            p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
            arr_clip = arr[(arr >= p1) & (arr <= p99)]
            ax.hist(arr_clip, bins=50, color=color, edgecolor="white", alpha=0.85)
            ax.set_title(f"{name}\nmean={arr.mean():.4f}  std={arr.std():.4f}", fontsize=11, fontweight="bold")
            ax.set_xlabel(name)
            ax.set_ylabel("Count")

    plt.suptitle("C4 — Distribusi Fitur Glottal (Voiced Frames)\nKorelasi kuat dengan kondisi depresi",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "c4_glottal_features.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")


# ── C5: Statistik agregat per partisipan ─────────────────────────────────────
def c5_agregat_stats(all_pids):
    print("\n" + "═"*60)
    print("C5 — Statistik Agregat per Partisipan (Feature Vectors)")
    print("═"*60)
    print("  🔄  Memproses semua partisipan (bisa memakan waktu)...")

    records = []
    for pid in all_pids:
        df = load_covarep(pid)
        if df is None:
            continue
        row = {"participant_id": pid}
        for i, col in enumerate(df.columns):
            vals = df.iloc[:, i].replace(0, np.nan).dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"]  = vals.std()
                row[f"{col}_min"]  = vals.min()
                row[f"{col}_max"]  = vals.max()
        records.append(row)

    df_agg = pd.DataFrame(records)
    out_path = os.path.join(OUT, "c5_feature_vectors.csv")
    df_agg.to_csv(out_path, index=False)
    print(f"  ✅  Feature vectors tersimpan: {df_agg.shape[0]} partisipan × {df_agg.shape[1]} kolom")
    print(f"  📄  Data disimpan → {out_path}")

    return df_agg


# ── C6: Deteksi file COVAREP corrupt (semua nol) ─────────────────────────────
def c6_detect_corrupt(all_pids):
    print("\n" + "═"*60)
    print("C6 — Deteksi File COVAREP Corrupt (Semua Nilai Nol)")
    print("═"*60)

    corrupts = []
    for pid in all_pids:
        df = load_covarep(pid)
        if df is None:
            corrupts.append((pid, "File tidak ditemukan"))
            continue
        if (df.values == 0).all():
            corrupts.append((pid, "Semua nilai nol"))

    if not corrupts:
        print("  ✅  Tidak ada file COVAREP yang corrupt.")
    else:
        print(f"  ⚠️  {len(corrupts)} file bermasalah:")
        for pid, reason in corrupts:
            print(f"      [{pid}]  {reason}")

    log_path = os.path.join(OUT, "c6_corrupt_check.txt")
    with open(log_path, "w") as f:
        if not corrupts:
            f.write("Semua file COVAREP normal.\n")
        else:
            for pid, reason in corrupts:
                f.write(f"{pid}: {reason}\n")
    print(f"  📄  Log disimpan → {log_path}")


# ── C7: Time-series F0 & Energy (1 sesi) ─────────────────────────────────────
def c7_timeseries(sample_pid):
    print("\n" + "═"*60)
    print(f"C7 — Time-Series F0 & Energy (Partisipan {sample_pid})")
    print("═"*60)

    df = load_covarep(sample_pid)
    if df is None:
        print(f"  ❌  File tidak ditemukan untuk {sample_pid}")
        return

    frame_ms = np.arange(len(df)) * 10  # ~10ms per frame

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # F0
    axes[0].plot(frame_ms / 1000, df.iloc[:, IDX_F0], color="#E91E63", linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel("F0 (Hz)", fontsize=11)
    axes[0].set_title(f"Partisipan {sample_pid} — Pitch (F0)", fontsize=11)

    # MCEP_0 (energy proxy)
    axes[1].plot(frame_ms / 1000, df.iloc[:, IDX_MCEP0], color="#2196F3", linewidth=0.5, alpha=0.8)
    axes[1].set_ylabel("MCEP_0 (Energy)", fontsize=11)
    axes[1].set_title(f"Partisipan {sample_pid} — Energy (MCEP_0)", fontsize=11)

    # VUV
    axes[2].fill_between(frame_ms / 1000, df.iloc[:, IDX_VUV], color="#4CAF50", alpha=0.6)
    axes[2].set_ylabel("Voiced (1) / Unvoiced (0)", fontsize=11)
    axes[2].set_title(f"Partisipan {sample_pid} — Voiced/Unvoiced Decision", fontsize=11)
    axes[2].set_xlabel("Waktu (detik)", fontsize=11)

    plt.suptitle(f"C7 — Time-Series Fitur Akustik\nPartisipan {sample_pid}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, f"c7_timeseries_{sample_pid}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EDA - C: EKSPLORASI COVAREP.csv")
    print("  Dataset : DAIC-WOZ")
    print("="*60)

    all_pids = get_participant_ids()
    samples  = random.sample(all_pids, min(SAMPLE_N, len(all_pids)))
    print(f"  Total partisipan : {len(all_pids)}")
    print(f"  Sampel analisis  : {samples}")

    c1_dimensi(samples)
    c2_silent_frames(all_pids)
    c3_pitch_distribution(samples)
    c4_glottal_features(samples)
    c5_agregat_stats(all_pids)
    c6_detect_corrupt(all_pids)
    c7_timeseries(samples[0])

    print("\n" + "="*60)
    print("  [OK] Selesai. Output tersimpan di:")
    print(f"  {OUT}")
    print("="*60 + "\n")
