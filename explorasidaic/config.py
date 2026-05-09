"""
config.py — Konfigurasi path & konstanta global untuk semua script EDA.
Ubah DATASET_DIR jika path dataset berbeda.
"""

import os

# ─── PATH UTAMA ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset", "raw", "DAIC-WOZ")
OUTPUT_DIR   = os.path.join(BASE_DIR, "explorasi", "output")

# ─── NAMA FILE PER PARTISIPAN ──────────────────────────────────────────────────
FILE_WAV        = "{pid}_AUDIO.wav"
FILE_COVAREP    = "{pid}_COVAREP.csv"
FILE_FORMANT    = "{pid}_FORMANT.csv"
FILE_TRANSCRIPT = "{pid}_TRANSCRIPT.csv"

# ─── MAPPING KOLOM COVAREP (74 kolom, berdasarkan COVAREP 2014 docs) ──────────
COVAREP_COLS = [
    "F0",          # 0  — Fundamental frequency (pitch), 0 = unvoiced
    "VUV",         # 1  — Voiced/Unvoiced decision (1=voiced, 0=unvoiced)
    "NAQ",         # 2  — Normalized Amplitude Quotient
    "QOQ",         # 3  — Quasi-Open Quotient
    "H1H2",        # 4  — Difference of 1st & 2nd harmonic amplitudes
    "PSP",         # 5  — Parabolic Spectral Parameter
    "MDQ",         # 6  — Maximum Dispersion Quotient
    "peakSlope",   # 7  — Peak Slope
    "Rd",          # 8  — Rd parameter of the LF glottal model
    "Rd_conf",     # 9  — Confidence of Rd
    "MCEP_0",      # 10 — MCEP coefficient 0
    *[f"MCEP_{i}" for i in range(1, 25)],  # 11-34: MCEP 1-24
    *[f"HMPDM_{i}" for i in range(25)],    # 35-59: HMPDM
    *[f"HMPDD_{i}" for i in range(13)],    # 60-72: HMPDD
    "shimmer",     # 73 — Shimmer
]

# ─── JUMLAH SAMPEL UNTUK ANALISIS BERAT (B, C, D, G) ─────────────────────────
SAMPLE_N = 10   # Default 10 partisipan acak untuk plot berat
RANDOM_SEED = 42
