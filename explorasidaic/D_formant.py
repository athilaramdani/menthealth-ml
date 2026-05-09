"""
D_formant.py — Eksplorasi D: Analisis Formant Frequencies (FORMANT.csv)
Mencakup item D1-D4 dari listexplorasi.md
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
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorasi.config import (DATASET_DIR, OUTPUT_DIR, FILE_FORMANT, SAMPLE_N, RANDOM_SEED)

OUT = os.path.join(OUTPUT_DIR, "D_formant")
os.makedirs(OUT, exist_ok=True)
random.seed(RANDOM_SEED)

# Kolom FORMANT.csv: [F1_freq, F1_bw, F2_freq, F2_bw, F3_freq, F3_bw, F4_freq, F4_bw]
FORMANT_COLS = ["F1_freq", "F1_bw", "F2_freq", "F2_bw",
                "F3_freq", "F3_bw", "F4_freq", "F4_bw"]


def get_participant_ids():
    pids = []
    for name in sorted(os.listdir(DATASET_DIR)):
        if os.path.isdir(os.path.join(DATASET_DIR, name)) and name.endswith("_P"):
            pids.append(name.replace("_P", ""))
    return pids


def load_formant(pid):
    path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_FORMANT.format(pid=pid))
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, header=None)
    n = min(df.shape[1], len(FORMANT_COLS))
    df.columns = FORMANT_COLS[:n]
    return df


# ── D1: Struktur kolom & sample baris pertama ────────────────────────────────
def d1_struktur(samples):
    print("\n" + "═"*60)
    print("D1 — Struktur FORMANT.csv")
    print("═"*60)

    for pid in samples[:3]:
        df = load_formant(pid)
        if df is None:
            print(f"  [{pid}]  File tidak ditemukan.")
            continue
        print(f"\n  [{pid}]  Shape={df.shape}")
        print(df.head(3).to_string(index=False))

    log_path = os.path.join(OUT, "d1_struktur.txt")
    with open(log_path, "w") as f:
        for pid in samples[:5]:
            df = load_formant(pid)
            if df is None:
                f.write(f"{pid}: File tidak ditemukan\n")
                continue
            f.write(f"\n[{pid}]  Shape={df.shape}\n")
            f.write(df.head(5).to_string() + "\n")
    print(f"\n  📄  Struktur disimpan → {log_path}")


# ── D2: Distribusi F1 dan F2 ─────────────────────────────────────────────────
def d2_distribusi_f1_f2(samples):
    print("\n" + "═"*60)
    print("D2 — Distribusi F1 dan F2")
    print("═"*60)

    all_f1, all_f2 = [], []
    for pid in samples:
        df = load_formant(pid)
        if df is None or "F1_freq" not in df.columns:
            continue
        f1 = df["F1_freq"].replace(0, np.nan).dropna()
        f2 = df["F2_freq"].replace(0, np.nan).dropna()
        all_f1.extend(f1.tolist())
        all_f2.extend(f2.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, name, color in zip(axes, [all_f1, all_f2], ["F1", "F2"], ["#E74C3C", "#3498DB"]):
        arr = np.array(data)
        p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        arr_clip = arr[(arr >= p1) & (arr <= p99)]
        ax.hist(arr_clip, bins=60, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(np.median(arr_clip), color="black", linestyle="--",
                   label=f"Median={np.median(arr_clip):.0f} Hz")
        ax.set_xlabel(f"{name} Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Distribusi {name} Formant Frequency", fontsize=12, fontweight="bold")
        ax.legend()

    plt.suptitle("D2 — Distribusi Formant F1 & F2\nBerkaitan dengan kualitas vokal dan vowel space",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "d2_distribusi_f1_f2.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")
    print(f"  F1 — median={np.median(all_f1):.1f} Hz  std={np.std(all_f1):.1f}")
    print(f"  F2 — median={np.median(all_f2):.1f} Hz  std={np.std(all_f2):.1f}")


# ── D3: Vowel Space Area (VSA) per partisipan ────────────────────────────────
def d3_vowel_space_area(samples):
    print("\n" + "═"*60)
    print("D3 — Vowel Space Area (VSA) per Partisipan")
    print("═"*60)
    print("  💡  VSA yang kecil berkorelasi dengan monotonnya suara (depresi).")

    records = []
    scatter_data = {}

    for pid in samples:
        df = load_formant(pid)
        if df is None or "F1_freq" not in df.columns:
            continue
        f1 = df["F1_freq"].replace(0, np.nan).dropna().values
        f2 = df["F2_freq"].replace(0, np.nan).dropna().values

        # Sejajarkan panjang
        n  = min(len(f1), len(f2))
        f1, f2 = f1[:n], f2[:n]
        pts = np.column_stack([f1, f2])

        # Filter outlier
        p5, p95 = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
        mask = np.all((pts >= p5) & (pts <= p95), axis=1)
        pts  = pts[mask]

        vsa = None
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                vsa  = hull.volume  # Di 2D, volume = area
            except Exception:
                vsa = None

        records.append({"participant_id": pid, "vsa": vsa})
        scatter_data[pid] = pts
        print(f"  [{pid}]  VSA = {vsa:.2f}" if vsa else f"  [{pid}]  VSA = N/A")

    df_vsa = pd.DataFrame(records)
    csv_path = os.path.join(OUT, "d3_vsa.csv")
    df_vsa.to_csv(csv_path, index=False)

    # Plot histogram VSA
    valid = df_vsa["vsa"].dropna()
    if len(valid) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(valid, bins=15, color="#8E44AD", edgecolor="white")
        ax.set_xlabel("Vowel Space Area", fontsize=12)
        ax.set_ylabel("Jumlah Partisipan", fontsize=12)
        ax.set_title("D3 — Distribusi Vowel Space Area (VSA)\nLebih kecil → cenderung lebih monoton",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(OUT, "d3_vsa_histogram.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\n  📊  Plot disimpan → {out_path}")
    print(f"  📄  Data disimpan → {csv_path}")


# ── D4: Korelasi formant antar kolom ─────────────────────────────────────────
def d4_korelasi_formant(samples):
    print("\n" + "═"*60)
    print("D4 — Korelasi antar Kolom Formant")
    print("═"*60)

    all_dfs = []
    for pid in samples:
        df = load_formant(pid)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("  ❌  Tidak ada data formant yang bisa dimuat.")
        return

    combined = pd.concat(all_dfs, ignore_index=True).replace(0, np.nan).dropna()
    corr = combined.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("D4 — Korelasi antar Fitur Formant\n(Semua sampel digabung)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "d4_korelasi_formant.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Heatmap disimpan → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EDA - D: EKSPLORASI FORMANT.csv")
    print("  Dataset : DAIC-WOZ")
    print("="*60)

    all_pids = get_participant_ids()
    samples  = random.sample(all_pids, min(SAMPLE_N, len(all_pids)))
    print(f"  Total partisipan : {len(all_pids)}")
    print(f"  Sampel analisis  : {samples}")

    d1_struktur(samples)
    d2_distribusi_f1_f2(samples)
    d3_vowel_space_area(samples)
    d4_korelasi_formant(samples)

    print("\n" + "="*60)
    print("  [OK] Selesai. Output tersimpan di:")
    print(f"  {OUT}")
    print("="*60 + "\n")
