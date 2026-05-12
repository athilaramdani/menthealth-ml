# Rencana Implementasi Pipeline DAIC-WOZ

**Peran**: ML & Data Engineer (Athila Ramdani Saputra)  
**Tujuan**: Implementasi full pipeline klasifikasi **3 kelas** kesehatan mental berbasis audio — DAIC-WOZ  
**Kelas Target**: `0 = Normal` | `1 = Kecemasan/Stress` | `2 = Depresi`  
**Referensi Pipeline**: `docs/pipelinedaic.png`

---

> ⚠️ **Catatan Labeling DAIC-WOZ**
>
> DAIC-WOZ **tidak memiliki label Kecemasan atau Stress secara eksplisit** — dataset ini dirancang
> khusus untuk deteksi Depresi (PHQ-8). Oleh karena itu, 3 kelas dibentuk menggunakan
> **PHQ-8 Severity sebagai proxy**:
>
> | Skor PHQ-8 | Label Kelas | Keterangan |
> | :--- | :--- | :--- |
> | 0 – 4  | `0 = Normal`            | Gejala minimal, tidak ada gangguan signifikan |
> | 5 – 14 | `1 = Kecemasan/Stress`  | Gejala ringan–sedang, proxy untuk distres psikologis |
> | ≥ 15   | `2 = Depresi`           | Gejala berat, depresi klinis |
>
> **Keterbatasan ini wajib dicantumkan sebagai *limitation* di laporan akhir.**
> Untuk klasifikasi 3 kelas dengan label eksplisit → gunakan dataset MODMA (PHQ-9 + GAD-7).



---

## Struktur File Output

Semua file disimpan di `experiments/`, menggunakan format `.py` → dikonversi ke `.ipynb` via `py_to_ipynb.py`.

```
experiments/
├── daic/
│   ├── Part1_Dataset_Overview.py         ← Bagian 1
│   ├── Part2_Preprocessing.py            ← Bagian 2
│   ├── Part3_Feature_Extraction.py       ← Bagian 3
│   ├── Part4_Dataset_Building.py         ← Bagian 4
│   ├── Part5_Split_Data.py               ← Bagian 5
│   ├── Part6_Training_Model.py           ← Bagian 6
│   └── Part7_XAI.py                      ← Bagian 7
dataset/
└── processed/
    ├── daic_metadata.csv                 ← Output Part 1
    ├── daic_features_raw.csv             ← Output Part 3
    ├── daic_features_final.csv           ← Output Part 4
    ├── daic_train.csv                    ← Output Part 5
    ├── daic_val.csv                      ← Output Part 5
    ├── daic_test.csv                     ← Output Part 5
    └── models/
        ├── best_model.pkl                ← Output Part 6
        └── scaler.pkl                    ← Output Part 6
```

---

## Rincian Setiap Part

### Part 1 — Dataset Overview
**File:** `experiments/daic/Part1_Dataset_Overview.py`  
**Isi:**
- Scan folder `dataset/raw/DAIC-WOZ/` dan inventori semua partisipan
- Baca label PHQ-8 dari file split (train/dev/test split CSV)
- Tampilkan distribusi label (Minimal/Mild/Moderate/Severe dan biner)
- Cek kelengkapan file per partisipan (AUDIO.wav, TRANSCRIPT.csv, COVAREP.csv)
- Output: `daic_metadata.csv` berisi `participant_id, phq8_score, label_biner, split`

---

### Part 2 — Preprocessing Audio
**File:** `experiments/daic/Part2_Preprocessing.py`  
**Isi:**
- Load `.wav` dengan `librosa` (sr=16000)
- Speaker Diarization sederhana: gunakan `TRANSCRIPT.csv` untuk filter hanya suara Participant (bukan Ellie)
- VAD (Voice Activity Detection): buang silence menggunakan `librosa.effects.split`
- Normalisasi amplitudo ke range [-1, 1]
- Simpan audio bersih sebagai array `.npy` atau langsung lanjut ke ekstraksi

---

### Part 3 — Ekstraksi Fitur Audio
**File:** `experiments/daic/Part3_Feature_Extraction.py`  
**Isi:**
- Ekstraksi per frame (window 25ms, hop 10ms):
  - MFCC (13 koefisien)
  - Pitch / F0 (via `librosa.piptrack`)
  - Energy / RMS (`librosa.feature.rms`)
  - Spectral Centroid, Bandwidth, Rolloff
  - Zero Crossing Rate
- Agregasi statistik per sesi: mean, std, min, max, percentile (25, 75)
- Output: `daic_features_raw.csv` (1 baris = 1 partisipan)

---

### Part 4 — Pembangunan Dataset (Dataset Building)
**File:** `experiments/daic/Part4_Dataset_Building.py`  
**Isi:**
- Merge `daic_features_raw.csv` + `daic_metadata.csv` berdasarkan `participant_id`
- Hapus fitur konstan atau highly correlated (threshold > 0.95)
- Seleksi fitur menggunakan ANOVA F-test / Mutual Information
- StandardScaler (fit HANYA pada train set)
- Output: `daic_features_final.csv`

---

### Part 5 — Split Data (Anti-Leakage)
**File:** `experiments/daic/Part5_Split_Data.py`  
**Isi:**
- Split berdasarkan Participant ID (BUKAN per segment)
- Gunakan split resmi DAIC-WOZ: `train_split_Depression_AVEC2017.csv`
- Validasi: pastikan tidak ada participant ID yang overlap antara train/val/test
- Proporsi: 70% train / 15% val / 15% test
- Setup GroupKFold untuk cross-validation
- Output: `daic_train.csv`, `daic_val.csv`, `daic_test.csv`

---

### Part 6 — Training Model
**File:** `experiments/daic/Part6_Training_Model.py`  
**Isi:**
- Model yang dilatih:
  1. Logistic Regression (baseline, `multi_class='auto'`)
  2. SVM (RBF kernel, `decision_function_shape='ovr'`)
  3. Random Forest
  4. XGBoost (`objective='multi:softmax'`, `num_class=3`)
- Proses:
  - GroupKFold Cross-Validation (5-fold, anti-leakage)
  - GridSearchCV dengan `scoring='f1_macro'`
  - Handle imbalance dengan `class_weight='balanced'`
- Evaluasi **Multiclass**:
  - Macro F1, Weighted F1, Accuracy, Precision, Recall
  - Confusion Matrix **3×3** (Stress | Kecemasan | Depresi)
  - `classification_report` per kelas
- Simpan model terbaik: `.pkl`
- Output: tabel perbandingan semua model

---

### Part 7 — XAI (Explainable AI)
**File:** `experiments/daic/Part7_XAI.py`  
**Isi:**
- SHAP (untuk RF & XGBoost) **Multiclass**:
  - Global: beeswarm plot per 3 kelas + bar chart mean |SHAP| gabungan
  - Local: waterfall plot per kelas (1 sampel per kelas)
  - Top 10 fitur per kelas
- LIME (untuk SVM & LR) **Multiclass**:
  - `top_labels=3` untuk semua kelas
  - 1 sampel per kelas aktual
  - Auto-install `lime` jika belum tersedia
- Visualisasi tambahan:
  - Prediksi vs Aktual scatter (3 kelas, color-coded)

---

## Urutan Pengerjaan

| Part | Status | Prioritas |
| :--- | :--- | :--- |
| Part 1 — Dataset Overview | ✅ Selesai | 🔴 Kerjakan Pertama |
| Part 2 — Preprocessing | ✅ Selesai | 🔴 Kerjakan Kedua |
| Part 3 — Feature Extraction | ✅ Selesai | 🔴 Kerjakan Ketiga |
| Part 4 — Dataset Building | ✅ Selesai | 🟡 Kerjakan Keempat |
| Part 5 — Split Data | ✅ Selesai | 🟡 Kerjakan Kelima |
| Part 6 — Training Model | ✅ Selesai | 🟢 Kerjakan Keenam |
| Part 7 — XAI | ✅ Selesai | 🟢 Kerjakan Ketujuh |

---

## Revisi: Perubahan ke Klasifikasi 3 Kelas

**Tanggal Revisi:** 2026-05-09  
**Alasan:** Proyek mengharuskan klasifikasi 3 kelas: **Depresi | Kecemasan | Stress** (sesuai notulensi rapat 28 April 2026).

### Labeling 3 Kelas Final (PHQ-8 Severity Proxy)

| Skor PHQ-8 | Kelas | Label |
| :--- | :---: | :--- |
| 0 – 4 | `0` | **Stress** (gejala minimal, stres sehari-hari) |
| 5 – 14 | `1` | **Kecemasan** (gejala ringan–sedang, distres/ansietas) |
| ≥ 15 | `2` | **Depresi** (gejala berat, depresi klinis) |

> ⚠️ Label Kecemasan & Stress adalah **proxy dari PHQ-8** — bukan label klinis eksplisit.  
> DAIC-WOZ hanya memiliki skor PHQ-8 (depresi). Keterbatasan ini wajib dicantumkan  
> sebagai *limitation* di laporan akhir. Untuk 3 kelas eksplisit → gunakan **MODMA** (PHQ-9 + GAD-7).

### Yang Diubah di Setiap Part

| Part | Perubahan |
| :--- | :--- |
| **Part 1** | `label_3kelas` menggantikan `label_biner`; warna baru; 2 threshold di histogram (PHQ-8=5 dan PHQ-8=15) |
| **Part 4** | Feature selection pakai `label_3kelas` (ANOVA & MI sudah native multiclass) |
| **Part 5** | `StratifiedShuffleSplit` pakai `label_3kelas`; bar chart distribusi 3 kelas per split |
| **Part 6** | Scoring `f1_macro`; SVM `decision_function_shape='ovr'`; XGBoost `objective='multi:softmax'`; confusion matrix 3×3; `classification_report` per kelas |
| **Part 7** | SHAP beeswarm per 3 kelas; waterfall per kelas; LIME `top_labels=3`; auto-install `lime` jika belum ada |

---

*Dibuat oleh: Athila Ramdani Saputra | ML & Data Engineer*  
*Terakhir diperbarui: 2026-05-09*
