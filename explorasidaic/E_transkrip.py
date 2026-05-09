"""
E_transkrip.py — Eksplorasi E: Analisis Transkrip Percakapan (TRANSCRIPT.csv)
Mencakup item E1-E8 dari listexplorasi.md
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorasi.config import (DATASET_DIR, OUTPUT_DIR, FILE_TRANSCRIPT)

OUT = os.path.join(OUTPUT_DIR, "E_transkrip")
os.makedirs(OUT, exist_ok=True)


def get_participant_ids():
    pids = []
    for name in sorted(os.listdir(DATASET_DIR)):
        if os.path.isdir(os.path.join(DATASET_DIR, name)) and name.endswith("_P"):
            pids.append(name.replace("_P", ""))
    return pids


def load_transcript(pid):
    path = os.path.join(DATASET_DIR, f"{pid}_P", FILE_TRANSCRIPT.format(pid=pid))
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        # Normalisasi nama kolom
        df = df.rename(columns={
            "start_time": "start_time",
            "stop_time":  "stop_time",
            "speaker":    "speaker",
            "value":      "value",
        })
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["stop_time"]  = pd.to_numeric(df["stop_time"],  errors="coerce")
        df["value"]      = df["value"].astype(str).str.strip()
        return df
    except Exception as e:
        return None


# ── E1-E8: Proses semua partisipan → ringkasan ──────────────────────────────
def e_proses_semua(all_pids):
    print("\n" + "═"*60)
    print("E1-E8 — Analisis Transkrip Seluruh Partisipan")
    print("═"*60)
    print("  🔄  Memproses semua transkrip...")

    records = []
    skip    = 0

    for pid in all_pids:
        df = load_transcript(pid)
        if df is None:
            skip += 1
            continue

        # E1: Filter hanya Participant
        participant_rows = df[df["speaker"].str.strip().str.lower() == "participant"].copy()
        ellie_rows       = df[df["speaker"].str.strip().str.lower() == "ellie"].copy()

        if participant_rows.empty:
            skip += 1
            continue

        participant_rows = participant_rows.dropna(subset=["start_time", "stop_time"])
        participant_rows["duration"] = participant_rows["stop_time"] - participant_rows["start_time"]
        participant_rows = participant_rows[participant_rows["duration"] > 0]

        # E2: Total durasi bicara
        total_speech = participant_rows["duration"].sum()

        # E3: Jumlah giliran bicara (turns)
        turn_count = len(participant_rows)

        # E4: Rata-rata durasi per giliran
        avg_turn_dur = participant_rows["duration"].mean() if turn_count > 0 else 0

        # E5: Total kata
        word_counts = participant_rows["value"].apply(
            lambda x: len(str(x).split()) if str(x).lower() not in ["nan", "", "[laughter]"] else 0
        )
        total_words = word_counts.sum()

        # E6: Speech rate (kata per detik)
        speech_rate = total_words / total_speech if total_speech > 0 else 0

        # E7: Pause ratio — total durasi sesi vs total bicara Participant
        sesi_start = df["start_time"].min()
        sesi_end   = df["stop_time"].max()
        sesi_dur   = sesi_end - sesi_start
        pause_ratio = 1 - (total_speech / sesi_dur) if sesi_dur > 0 else np.nan

        records.append({
            "participant_id":      pid,
            "total_speech_sec":    round(total_speech, 2),
            "turn_count":          turn_count,
            "total_words":         int(total_words),
            "avg_turn_duration":   round(avg_turn_dur, 2),
            "speech_rate_wps":     round(speech_rate, 3),
            "pause_ratio":         round(pause_ratio, 3) if not np.isnan(pause_ratio) else None,
            "session_duration_sec": round(sesi_dur, 2),
        })

    df_summary = pd.DataFrame(records)
    print(f"\n  Total diproses     : {len(df_summary)} partisipan")
    print(f"  Skip (error/kosong): {skip}")
    print(f"\n  ─── Statistik Ringkasan ───")
    for col in ["total_speech_sec", "turn_count", "total_words", "speech_rate_wps", "pause_ratio"]:
        if col in df_summary.columns:
            s = df_summary[col].dropna()
            print(f"  {col:<25} mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")

    # Simpan E8 — Tabel ringkasan
    csv_path = os.path.join(OUT, "e8_ringkasan_transkrip.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\n  📄  Tabel E8 disimpan → {csv_path}")

    return df_summary


# ── Plot distribusi fitur transkrip ──────────────────────────────────────────
def e_plot_distribusi(df_summary):
    print("\n" + "─"*60)
    print("  Membuat plot distribusi fitur transkrip...")

    features = [
        ("total_speech_sec",  "Total Durasi Bicara (detik)",    "#2196F3"),
        ("turn_count",        "Jumlah Giliran Bicara (Turns)",  "#4CAF50"),
        ("total_words",       "Total Kata Diucapkan",           "#FF9800"),
        ("speech_rate_wps",   "Speech Rate (kata/detik)",       "#9C27B0"),
        ("pause_ratio",       "Pause Ratio",                    "#F44336"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (col, label, color) in enumerate(features):
        data = df_summary[col].dropna()
        ax   = axes[i]
        ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean={data.mean():.2f}")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Jumlah Partisipan")
        ax.legend(fontsize=8)

    axes[-1].set_visible(False)
    plt.suptitle("E — Distribusi Fitur Percakapan dari Transkrip DAIC-WOZ\n(Semua Partisipan)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "e_distribusi_fitur_transkrip.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Plot disimpan → {out_path}")


# ── Plot scatter speech rate vs pause ratio ──────────────────────────────────
def e_plot_scatter(df_summary):
    print("  Membuat scatter plot speech rate vs pause ratio...")

    df_clean = df_summary.dropna(subset=["speech_rate_wps", "pause_ratio"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df_clean["speech_rate_wps"], df_clean["pause_ratio"],
               alpha=0.6, color="#673AB7", edgecolors="white", s=60)
    ax.set_xlabel("Speech Rate (kata/detik)", fontsize=12)
    ax.set_ylabel("Pause Ratio", fontsize=12)
    ax.set_title("E — Speech Rate vs Pause Ratio\nOrang depresi cenderung kanan-atas (lambat & banyak jeda)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "e_scatter_speechrate_pause.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊  Scatter disimpan → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EDA - E: EKSPLORASI TRANSKRIP PERCAKAPAN")
    print("  Dataset : DAIC-WOZ")
    print("="*60)

    all_pids = get_participant_ids()
    print(f"  Total partisipan: {len(all_pids)}")

    df_summary = e_proses_semua(all_pids)
    e_plot_distribusi(df_summary)
    e_plot_scatter(df_summary)

    print("\n" + "="*60)
    print("  [OK] Selesai. Output tersimpan di:")
    print(f"  {OUT}")
    print("="*60 + "\n")
