# Eksplorasi Data Audio MODMA

Dokumen ini berisi hasil Eksplorasi Data Analysis (EDA) khusus untuk dataset audio MODMA. Proses analisis dijalankan menggunakan skrip `H_modma_audio.py` di dalam direktori `explorasi`.

## 1. Distribusi Dataset & Durasi

Berbeda dengan dataset DAIC-WOZ di mana 1 partisipan umumnya memiliki 1 file audio wawancara panjang, pada MODMA setiap partisipan merekam banyak file pendek.

- **Total Partisipan**: 52 Partisipan (29 HC, 23 MDD)
- **Rata-rata Jumlah File/Partisipan**: Tepat 29 file `.wav` baik untuk HC maupun MDD.
- **Rata-rata Total Durasi/Partisipan**:
  - HC: ~496.3 detik (~8.2 menit)
  - MDD: ~499.1 detik (~8.3 menit)
- **Rata-rata Durasi Per File Rekaman**: ~17 detik (baik HC maupun MDD).

> [!TIP]
> Fakta bahwa jumlah file secara konsisten adalah 29 file per partisipan menandakan adanya *structured task* (tugas membaca/berbicara terstruktur) saat pengambilan data MODMA (berbeda dengan DAIC-WOZ yang bersifat percakapan bebas / wawancara). Total durasi antara kelompok MDD dan HC sangat seimbang.

### Visualisasi Distribusi Jumlah dan Durasi

![Distribusi Jumlah File dan Durasi](./assets/modma/h1_distribusi_jumlah_dan_durasi.png)

## 2. Karakteristik Audio (Sampling Rate)

- **Total File Audio**: 1.503 file
- **Sampling Rate**: Semua file (100%) memiliki sampling rate **44.100 Hz** (44.1 kHz). 

> [!NOTE]
> Kualitas audio sangat konsisten. Saat kita akan menggabungkannya ke model pipeline, pastikan kita mendownsample ke standar yang sama dengan DAIC-WOZ jika kita akan memodelkan kedua dataset sekaligus (biasanya 16kHz sudah cukup untuk deteksi suara/speech), atau biarkan 44.1kHz jika kita hanya melatih model di atas MODMA saja.

## 3. Sampel Visualisasi Waveform & Spectrogram

Berikut adalah contoh pengambilan secara acak cuplikan audio selama 10 detik dari 1 penderita Major Depressive Disorder (MDD) dan 1 Healthy Control (HC).

![Sampel Waveform dan Spectrogram](./assets/modma/h3_waveform_spectrogram_samples.png)

## Kesimpulan Awal

1. Dataset sudah sangat rapi dan siap untuk diproses ke tahap ekstraksi fitur (MFCC, Covarep, dsb.).
2. Karena formatnya terpecah-pecah (29 file per partisipan), saat ekstraksi fitur kita memiliki dua opsi strategi:
   - **Opsi A**: Mengekstraksi fitur per file `.wav`, lalu merata-ratakannya per partisipan.
   - **Opsi B**: Menggabungkan (*concatenate*) ke-29 file tersebut menjadi satu file `.wav` utuh sebelum diekstrak fiturnya (menyerupai gaya DAIC-WOZ).

## 4. Metadata Labeling Tambahan

Berdasarkan analisis file `subjects_information_audio_lanzhou_2015.xlsx`, dataset MODMA tidak hanya menyediakan label diagnosis biner (MDD vs HC), tetapi juga label kuantitatif dari berbagai kuesioner psikologi:

1. **Depresi (Depression):**
   - **`type`**: Label diagnostik (MDD atau HC).
   - **`PHQ-9`**: Patient Health Questionnaire-9 (skor 0-25).
2. **Kecemasan (Anxiety):**
   - **`GAD-7`**: Generalized Anxiety Disorder-7 (skor 0-21).
3. **Stres & Trauma (Stress):**
   - **`LES`**: Life Events Scale (tingkat stres akibat peristiwa kehidupan).
   - **`CTQ-SF`**: Childhood Trauma Questionnaire (trauma masa lalu).
4. **Kualitas Tidur:**
   - **`PSQI`**: Pittsburgh Sleep Quality Index.

Fitur ini membuka peluang untuk eksperimen pemodelan *regression* guna memprediksi skor tingkat keparahan (seperti memprediksi tingkat kecemasan / depresi dalam nilai numerik), bukan hanya sekadar membedakan kelas pasien.
