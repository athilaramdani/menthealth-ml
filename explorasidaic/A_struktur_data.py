"""
A_struktur_data.py — Eksplorasi A: Struktur & Integritas Dataset DAIC-WOZ
Mencakup item A1, A2, A3, A4, A5 dari listexplorasi.md
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explorasi.config import (DATASET_DIR, OUTPUT_DIR,
                               FILE_WAV, FILE_COVAREP, FILE_FORMANT, FILE_TRANSCRIPT)

OUT = os.path.join(OUTPUT_DIR, "A_struktur")
os.makedirs(OUT, exist_ok=True)

REQUIRED_FILES = [FILE_WAV, FILE_COVAREP, FILE_FORMANT, FILE_TRANSCRIPT]
LABEL_FILENAMES = [
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression.csv",
    "test_split_Depression.csv",
    "labels.csv", "label.csv",
]

# ══════════════════════════════════════════════════════════════════════════════
def get_participant_folders():
    """Kembalikan list (pid_str, folder_path) untuk semua folder _P."""
    folders = []
    for name in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, name)
        if os.path.isdir(path) and name.endswith("_P"):
            pid = name.replace("_P", "")
            folders.append((pid, path))
    return folders


# ── A1: Total partisipan ──────────────────────────────────────────────────────
def a1_total_partisipan(folders):
    print("\n" + "═"*60)
    print("A1 — Total Partisipan")
    print("═"*60)
    print(f"  Total folder partisipan ditemukan : {len(folders)}")
    ids = [int(pid) for pid, _ in folders]
    print(f"  ID terkecil                       : {min(ids)}")
    print(f"  ID terbesar                       : {max(ids)}")


# ── A2: Missing file per partisipan ──────────────────────────────────────────
def a2_missing_files(folders):
    print("\n" + "═"*60)
    print("A2 — Cek File Hilang per Partisipan")
    print("═"*60)

    missing_log = []
    for pid, folder in folders:
        missing = []
        for tmpl in REQUIRED_FILES:
            fname = tmpl.format(pid=pid)
            if not os.path.isfile(os.path.join(folder, fname)):
                missing.append(fname)
        if missing:
            missing_log.append((pid, missing))

    if not missing_log:
        print("  ✅  Semua partisipan memiliki file lengkap.")
    else:
        print(f"  ⚠️  {len(missing_log)} partisipan memiliki file yang hilang:\n")
        for pid, files in missing_log:
            print(f"    [{pid}_P]  Missing: {', '.join(files)}")

    # Simpan log
    log_path = os.path.join(OUT, "a2_missing_files.txt")
    with open(log_path, "w") as f:
        if not missing_log:
            f.write("Semua file lengkap.\n")
        else:
            for pid, files in missing_log:
                f.write(f"{pid}_P: {', '.join(files)}\n")
    print(f"\n  📄  Log disimpan → {log_path}")


# ── A3: Nomor partisipan yang hilang dalam urutan ────────────────────────────
def a3_id_gap(folders):
    print("\n" + "═"*60)
    print("A3 — Nomor Partisipan Hilang (Gap dalam Urutan)")
    print("═"*60)

    existing_ids = set(int(pid) for pid, _ in folders)
    id_min, id_max = min(existing_ids), max(existing_ids)
    full_range = set(range(id_min, id_max + 1))
    missing_ids = sorted(full_range - existing_ids)

    print(f"  Rentang urutan penuh : {id_min} – {id_max}")
    print(f"  Total ID yang ada    : {len(existing_ids)}")
    print(f"  Total ID hilang      : {len(missing_ids)}")
    if missing_ids:
        print(f"  ID hilang            : {missing_ids}")

    log_path = os.path.join(OUT, "a3_id_gap.txt")
    with open(log_path, "w") as f:
        f.write(f"Rentang: {id_min}-{id_max}\n")
        f.write(f"ID hilang ({len(missing_ids)}): {missing_ids}\n")
    print(f"  📄  Log disimpan → {log_path}")


# ── A4: Cari file label PHQ-8 ────────────────────────────────────────────────
def a4_cari_label():
    print("\n" + "═"*60)
    print("A4 — [KRITIS] Pencarian File Label PHQ-8")
    print("═"*60)

    found = []
    search_dirs = [
        DATASET_DIR,
        os.path.join(DATASET_DIR, "util"),
        os.path.join(DATASET_DIR, "documents"),
    ]
    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        for name in os.listdir(sdir):
            if name.lower().endswith(".csv"):
                found.append(os.path.join(sdir, name))
            for kw in ["label", "split", "phq", "avec", "train", "dev", "test"]:
                if kw in name.lower():
                    found.append(os.path.join(sdir, name))

    found = list(set(found))
    label_candidates = [f for f in found if any(
        kw in os.path.basename(f).lower()
        for kw in ["label", "split", "phq", "avec"]
    )]

    if label_candidates:
        print(f"  ✅  File label kandidat ditemukan:")
        for f in label_candidates:
            print(f"      {f}")
    else:
        print("  ❌  TIDAK ADA file label PHQ-8 ditemukan di dataset.")
        print("  ⚠️  Tindakan: Apply ke USC ICT atau tanyakan ke mentor.")

    log_path = os.path.join(OUT, "a4_label_check.txt")
    with open(log_path, "w") as f:
        f.write("File CSV ditemukan di dataset:\n")
        for fp in found:
            f.write(f"  {fp}\n")
        f.write("\nLabel kandidat:\n")
        for fp in label_candidates:
            f.write(f"  {fp}\n")
        if not label_candidates:
            f.write("  [TIDAK ADA]\n")
    print(f"  📄  Log disimpan → {log_path}")


# ── A5: Verifikasi format WAV ─────────────────────────────────────────────────
def a5_format_wav(folders):
    print("\n" + "═"*60)
    print("A5 — Verifikasi Format File .wav")
    print("═"*60)

    sr_counts  = {}
    ch_counts  = {}
    fmt_counts = {}
    errors     = []

    for pid, folder in folders:
        wav_path = os.path.join(folder, FILE_WAV.format(pid=pid))
        if not os.path.isfile(wav_path):
            continue
        try:
            info = sf.info(wav_path)
            sr_counts[info.samplerate]  = sr_counts.get(info.samplerate, 0) + 1
            ch_counts[info.channels]    = ch_counts.get(info.channels, 0) + 1
            fmt_counts[info.subtype]    = fmt_counts.get(info.subtype, 0) + 1
        except Exception as e:
            errors.append((pid, str(e)))

    print(f"  Sample Rate  : {dict(sorted(sr_counts.items()))}")
    print(f"  Channels     : {dict(sorted(ch_counts.items()))}")
    print(f"  Format/Subtype : {dict(sorted(fmt_counts.items()))}")
    if errors:
        print(f"\n  ❌  Error baca {len(errors)} file:")
        for pid, err in errors:
            print(f"      [{pid}_P] {err}")
    else:
        print("  ✅  Semua file WAV berhasil dibaca.")

    log_path = os.path.join(OUT, "a5_wav_format.txt")
    with open(log_path, "w") as f:
        f.write(f"Sample Rate  : {sr_counts}\n")
        f.write(f"Channels     : {ch_counts}\n")
        f.write(f"Format       : {fmt_counts}\n")
        if errors:
            f.write(f"\nError:\n")
            for pid, err in errors:
                f.write(f"  {pid}_P: {err}\n")
    print(f"  📄  Log disimpan → {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EDA - A: EKSPLORASI STRUKTUR & INTEGRITAS DATA")
    print("  Dataset : DAIC-WOZ")
    print("="*60)

    folders = get_participant_folders()

    a1_total_partisipan(folders)
    a2_missing_files(folders)
    a3_id_gap(folders)
    a4_cari_label()
    a5_format_wav(folders)

    print("\n" + "="*60)
    print("  [OK] Selesai. Output tersimpan di:")
    print(f"  {OUT}")
    print("="*60 + "\n")
