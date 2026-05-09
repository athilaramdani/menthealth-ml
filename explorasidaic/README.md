# Eksplorasi EDA — Dataset DAIC-WOZ

Folder ini berisi semua script Python untuk **Exploratory Data Analysis (EDA)** dataset DAIC-WOZ pada proyek Menthealth-ML.

## Struktur File

```
explorasi/
├── config.py               ← Konfigurasi path & konstanta global
├── run_all.py              ← Jalankan SEMUA section sekaligus
│
├── A_struktur_data.py      ← A: Integritas & kelengkapan dataset
├── B_audio_wav.py          ← B: Eksplorasi file audio .wav
├── C_covarep.py            ← C: Fitur akustik COVAREP (74 kolom)
├── D_formant.py            ← D: Formant frequencies & Vowel Space
├── E_transkrip.py          ← E: Analisis transkrip percakapan
└── G_fitur_lanjutan.py     ← G: MFCC, Spectral, ZCR, Jitter, Shimmer

output/                     ← Dibuat otomatis saat script dijalankan
├── A_struktur/
├── B_audio_wav/
├── C_covarep/
├── D_formant/
├── E_transkrip/
└── G_fitur_lanjutan/
```

## Cara Menjalankan

### Aktivasi Virtual Environment
```powershell
# dari root repo
.\.venv\Scripts\activate
```

### Jalankan Semua Sekaligus
```powershell
python explorasi/run_all.py
```

### Jalankan Per Section
```powershell
python explorasi/A_struktur_data.py
python explorasi/B_audio_wav.py
python explorasi/C_covarep.py
python explorasi/D_formant.py
python explorasi/E_transkrip.py
python explorasi/G_fitur_lanjutan.py
```

> ⚠️ **Section B & G** memakan waktu lebih lama karena harus load file audio `.wav`.
> Secara default hanya 10 sampel yang diproses untuk B & G (bisa diubah di `config.py`).

## Output per Section

| Section | File Output Utama | Deskripsi |
|---|---|---|
| A | `a2_missing_files.txt`, `a3_id_gap.txt`, `a4_label_check.txt`, `a5_wav_format.txt` | Integritas dataset |
| B | `b1_distribusi_durasi.png`, `b3_waveform_samples.png`, `b4_rms_energy.png`, `b6_mel_spectrogram.png` | Visualisasi audio |
| C | `c2_silent_frames.png`, `c3_pitch_distribution.png`, `c4_glottal_features.png`, `c5_feature_vectors.csv` | Fitur COVAREP |
| D | `d2_distribusi_f1_f2.png`, `d3_vsa_histogram.png`, `d4_korelasi_formant.png` | Formant & VSA |
| E | `e8_ringkasan_transkrip.csv`, `e_distribusi_fitur_transkrip.png`, `e_scatter_speechrate_pause.png` | Speech behavior |
| G | `g1_mfcc_samples.png`, `g2_g3_g4_spectral_features.png`, `g5_jitter_shimmer.png` | Fitur lanjutan |

## Konfigurasi (`config.py`)

```python
DATASET_DIR  = "dataset/raw/DAIC-WOZ"   # Path ke dataset
SAMPLE_N     = 10                         # Jumlah sampel untuk plot berat
RANDOM_SEED  = 42
```

## Catatan

- **Section F** (Eksplorasi Label) belum dibuat karena file label PHQ-8 belum tersedia. Akan ditambahkan setelah mendapat akses dari USC ICT atau mentor.
- **MODMA** dikecualikan karena dataset belum diunduh.
