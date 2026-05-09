# List Eksplorasi EDA — Dataset DAIC-WOZ

Dokumen ini berisi daftar lengkap item eksplorasi untuk tahap **Exploratory Data Analysis (EDA)** pada dataset **DAIC-WOZ** yang tersedia di `dataset/raw/DAIC-WOZ/`.

> **Catatan**: Dataset MODMA belum tersedia, sehingga EDA ini hanya mencakup DAIC-WOZ.
> Dataset DAIC-WOZ terdiri dari **189 partisipan** (ID 300–492), setiap partisipan memiliki file: `.wav`, `COVAREP.csv`, `FORMANT.csv`, dan `TRANSCRIPT.csv`.

---

## A. Eksplorasi Struktur & Integritas Data

- [ ] **A1** — Hitung total partisipan & verifikasi semua folder lengkap (tidak ada yang rusak/kosong).
- [ ] **A2** — Cek apakah ada partisipan yang *missing* file tertentu (contoh: ada `COVAREP.csv` tapi tidak ada `.wav`).
- [ ] **A3** — Identifikasi nomor partisipan yang hilang dalam urutan (misal: tidak ada folder `342_P`, `394_P`, `398_P`, `460_P`).
- [ ] **A4** — **[KRITIS]** Cari file label (PHQ-8 score / `train_split_Depression_AVEC2017.csv`) — tanpa ini training model tidak bisa dilakukan.
- [ ] **A5** — Verifikasi format semua file `.wav` (apakah mono/stereo, encoding, bit depth).

---

## B. Eksplorasi File Audio (`.wav`)

- [ ] **B1** — Tampilkan distribusi **durasi rekaman** (histogram) — berapa menit rata-rata per sesi wawancara.
- [ ] **B2** — Cek **sampling rate** semua file — pastikan konsisten (target: 16kHz untuk speech processing).
- [ ] **B3** — Plot **waveform** beberapa sampel partisipan secara acak untuk inspeksi visual.
- [ ] **B4** — Hitung **RMS Energy** rata-rata tiap file — apakah ada variasi signifikan antar partisipan?
- [ ] **B5** — Cek **clipping** (audio yang terlalu keras / pecah) — nilai amplitudo mendekati ±1.0.
- [ ] **B6** — Plot **Mel-Spectrogram** beberapa sampel untuk melihat pola frekuensi secara visual.

---

## C. Eksplorasi Fitur Akustik (`COVAREP.csv`)

> File `COVAREP.csv` berisi **74 kolom fitur** per frame (~10ms). Tidak ada header, kolom perlu di-mapping manual sesuai dokumentasi COVAREP 2014.

- [ ] **C1** — Tampilkan dimensi file (jumlah baris/frame & 74 kolom) untuk beberapa partisipan.
- [ ] **C2** — Hitung **persentase silent frames** (baris di mana kolom pertama `F0 = 0`) — orang depresi cenderung lebih sering diam.
- [ ] **C3** — Plot distribusi **F0 (Pitch)** hanya dari voiced frames (F0 > 0) — cek apakah pitch terlihat monoton atau variatif.
- [ ] **C4** — Plot distribusi **fitur glottal** (kolom NAQ, QOQ, H1-H2) — berkorelasi kuat dengan depresi.
- [ ] **C5** — Hitung **statistik agregat per partisipan** (mean, std, min, max) dari seluruh 74 fitur sebagai dasar feature vector untuk ML.
- [ ] **C6** — Cek apakah ada partisipan yang seluruh COVAREP-nya nol / corrupt.
- [ ] **C7** — Plot **time-series** beberapa fitur (F0, energy) dari satu sesi wawancara untuk memahami dinamika sinyal.

---

## D. Eksplorasi Formant (`FORMANT.csv`)

> Berisi frekuensi formant F1, F2, F3, F4 dan bandwidth-nya.

- [ ] **D1** — Tampilkan sample baris pertama — pahami struktur kolom (F1 freq, F1 bandwidth, F2 freq, dst).
- [ ] **D2** — Plot distribusi **F1 dan F2** — ini adalah fitur vokal yang membedakan kualitas bunyi (vowel space).
- [ ] **D3** — Hitung **Vowel Space Area** (VSA) dari F1 dan F2 — literatur menunjukkan VSA mengecil pada depresi.
- [ ] **D4** — Hitung korelasi formant dengan fitur COVAREP untuk memahami redundansi fitur.

---

## E. Eksplorasi Transkrip (`TRANSCRIPT.csv`)

> Berisi kolom: `start_time`, `stop_time`, `speaker` (Ellie/Participant), `value` (teks ucapan).

- [ ] **E1** — Pisahkan baris berdasarkan speaker — filter hanya ucapan **Participant** (bukan Ellie).
- [ ] **E2** — Hitung **total durasi bicara Participant** = sum(`stop_time - start_time`) per sesi.
- [ ] **E3** — Hitung **jumlah giliran bicara** (*speaking turns*) Participant per sesi.
- [ ] **E4** — Hitung **rata-rata panjang respons** Participant (durasi per giliran bicara).
- [ ] **E5** — Hitung **jumlah kata** yang diucapkan Participant (panjang string `value`).
- [ ] **E6** — Hitung **Speech Rate** (kata per detik) dari Participant.
- [ ] **E7** — Hitung **pause ratio** — perbandingan waktu diam vs waktu bicara Participant.
- [ ] **E8** — Buat tabel ringkasan per partisipan: `[participant_id, total_speech_duration, turn_count, word_count, speech_rate, pause_ratio]`.

> **Insight klinis**: Orang dengan depresi cenderung bicara lebih sedikit, lebih lambat, dan lebih banyak berdiam diri.

---

## F. Eksplorasi Label (Jika Sudah Tersedia)

> *Bagian ini dikerjakan setelah file label PHQ-8 ditemukan/diunduh.*

- [ ] **F1** — Load file label dan distribusikan ke kelas: **Depresi** (PHQ-8 ≥ 10) vs **Normal** (PHQ-8 < 10).
- [ ] **F2** — Hitung **class imbalance** — berapa persen depresi vs normal?
- [ ] **F3** — Plot **distribusi skor PHQ-8** (histogram continuous score).
- [ ] **F4** — Buat **boxplot** fitur-fitur kunci (pitch, energy, pause ratio) dikelompokkan per kelas — cek perbedaan yang terlihat.
- [ ] **F5** — Hitung **korelasi Pearson/Spearman** antara skor PHQ-8 dengan setiap fitur audio.
- [ ] **F6** — Identifikasi fitur **Top-10** yang paling berkorelasi dengan label depresi.

---

## G. Eksplorasi Khusus — Fitur Relevan dari PPT & Literatur

Berdasarkan slide PPT Pertemuan 0 dan notulensi mentor:

- [ ] **G1** — Ekstraksi **MFCC (13-40 koefisien)** langsung dari file `.wav` menggunakan `librosa` — bandingkan dengan yang ada di COVAREP.
- [ ] **G2** — Plot **Spectral Centroid** per sesi — menangkap "titik berat" frekuensi emosi.
- [ ] **G3** — Plot **Zero Crossing Rate (ZCR)** — berkorelasi dengan "kerasnya" atau "nyaringnya" suara.
- [ ] **G4** — Hitung **Spectral Rolloff** — frekuensi di mana 85% energi terkandung.
- [ ] **G5** — Hitung **Jitter & Shimmer** dari data COVAREP — indikator ketidakstabilan suara (tremor).

---

## Prioritas Pengerjaan

| Prioritas | Item | Alasan |
| --- | --- | --- |
| 🔴 **Kritis** | A4 (Cari Label), A1, A2 | Tanpa label, training tidak bisa dilakukan |
| 🟠 **Tinggi** | B1-B4, C2-C5, E1-E8 | Fondasi pemahaman data sebelum feature engineering |
| 🟡 **Sedang** | D1-D4, G1-G5 | Fitur tambahan yang relevan secara klinis |
| 🟢 **Opsional** | F1-F6 | Dilakukan setelah label tersedia |

---

## Tools yang Digunakan

```python
import librosa          # Audio loading, MFCC, Spectral features
import pandas as pd     # Membaca COVAREP.csv, FORMANT.csv, TRANSCRIPT.csv
import numpy as np      # Komputasi statistik
import matplotlib.pyplot as plt   # Visualisasi waveform & plot
import seaborn as sns   # Heatmap korelasi & distribusi fitur
```
