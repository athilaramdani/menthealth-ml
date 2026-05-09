# Analisis Dataset Menthealth-ML (DAIC-WOZ)

Berdasarkan eksplorasi pada direktori `d:\repositories\menthealth-ml\dataset\raw\DAIC-WOZ`, berikut adalah ringkasan dan analisis dataset yang tersedia.

## 1. Struktur Dataset
Dataset DAIC-WOZ (The Distress Analysis Interview Corpus Wizard-of-Oz) terdiri dari rekaman wawancara klinis yang dirancang untuk membantu diagnosis gangguan psikologis seperti depresi dan PTSD.

**Jumlah Sampel**: Terdeteksi sekitar **189 folder partisipan** (ID mulai dari 300 hingga 492).

### Isi Setiap Folder Partisipan (Contoh: `300_P`)
| Nama File | Ukuran | Deskripsi |
| --- | --- | --- |
| `300_AUDIO.wav` | ~20 MB | Rekaman audio asli (Raw Audio). |
| `300_TRANSCRIPT.csv` | ~8 KB | Teks percakapan antara Ellie (AI) dan Partisipan. |
| `300_COVAREP.csv` | ~35 MB | Fitur akustik (74 kolom) mencakup prosodi dan glottal. |
| `300_FORMANT.csv` | ~2 MB | Frekuensi formant (fitur vokal). |
| `300_CLNF_*.txt` | Variatif | Fitur visual (Action Units, Gaze, Pose) hasil OpenFace. |
| `300_CLNF_hog.txt` | ~347 MB | Histogram of Oriented Gradients (sangat besar). |

## 2. Analisis Konten
### Fitur Akustik (`COVAREP.csv`)
File ini berisi 74 fitur numerik yang diekstraksi per frame (biasanya per 10ms). Fitur ini mencakup:
- **F0 (Pitch)**: Nada dasar suara.
- **VUV**: Voiced/Unvoiced decision.
- **NAQ, QOQ, H1-H2**: Fitur glottal flow yang sering berkorelasi dengan tingkat ketegangan suara (indikator stres/depresi).
- **MCEP**: Mel-Cepstral Coefficients.

### Transkrip (`TRANSCRIPT.csv`)
Berisi label pembicara (`Ellie` atau `Participant`) beserta durasi waktu. Contoh dari partisipan 300:
- **Ellie**: *"have you been diagnosed with depression"*
- **Participant**: *"no"*

## 3. Temuan Penting & Masalah (Critical Issues)
> [!CAUTION]
> **Ground Truth Labels Tidak Ditemukan**
> Saya tidak menemukan file master CSV (seperti `train_split.csv` atau `test_split.csv`) yang berisi skor **PHQ-8** atau label biner depresi. Tanpa file ini, model Machine Learning tidak bisa dilatih (Supervised Learning).

> [!WARNING]
> **Kapasitas Penyimpanan**
> File `CLNF_hog.txt` berukuran sangat besar (~350MB per partisipan). Jika ditotal untuk ~189 partisipan, ini memakan ruang sekitar **60-70 GB**. Jika proyek Anda hanya berfokus pada **Audio-only**, file-file HOG dan fitur visual lainnya sebaiknya dihapus atau dipindahkan untuk menghemat ruang.

## 4. Rekomendasi Langkah Selanjutnya
1. **Cari File Label**: Pastikan Anda memiliki file split (biasanya dari kompetisi AVEC 2017) yang mencocokkan `Participant_ID` dengan label target.
2. **Preprocessing Audio**: Karena lingkup proyek adalah Machine Learning, Anda perlu melakukan agregasi fitur dari `COVAREP.csv` (misalnya menghitung mean, std, skewness per sesi) untuk mendapatkan satu vektor fitur per partisipan.
3. **Ekstraksi MFCC**: Jika `COVAREP` dirasa kurang mencukupi, Anda bisa mengekstrak MFCC tambahan langsung dari file `.wav` menggunakan library seperti `librosa`.
