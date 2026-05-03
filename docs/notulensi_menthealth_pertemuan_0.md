# NOTULENSI RAPAT MENTORING
**Proyek Menthealth — Klasifikasi Kesehatan Mental Berbasis Audio**
*Selasa, 28 April 2026 | 09.00 WIB | Meeting Online*

## 1. INFORMASI RAPAT

| Keterangan | Detail |
| --- | --- |
| Tanggal / Waktu | Selasa, 28 April 2026 / 09.00 WIB |
| Mentor | Muhammad Ridha |
| Proyek | Menthealth — Klasifikasi Kesehatan Mental Berbasis Audio |
| Peserta Hadir | 3 Orang Intern (Muhammad Mufid Taqiyuddin, Muhammad Raihan Thaffan Hidayat, Athila Ramdani Saputra) |
| Platform | Online (Video Conference) |
| Sifat Pertemuan | Onboarding & Kickoff Proyek Internship |

## 2. DAFTAR PESERTA

| No | Nama | Asal / Prodi | Posisi dalam Tim |
| --- | --- | --- | --- |
| 1 | Muhammad Mufid Taqiyuddin | Sistem Informasi — FRI | UI/UX Designer |
| 2 | Muhammad Raihan Thaffan Hidayat | Teknik Telekomunikasi | Front-End Developer |
| 3 | Athila Ramdani Saputra | (Prodi tidak disebutkan) | Back-End Developer & Project Manager |

📝 **Catatan**: Seluruh peserta hadir secara online. Beberapa peserta awalnya belum mengaktifkan kamera dan diminta untuk menyalakannya oleh mentor.

## 3. AGENDA PERTEMUAN

- Perkenalan mentor dan seluruh peserta intern
- Pemaparan gambaran besar proyek Menthealth
- Penjelasan dataset, fitur audio, pendekatan ML/DL, dan XAI
- Pemaparan arsitektur sistem dan website
- Presentasi timeline pengerjaan 2 bulan
- Penentuan Project Manager dan pembagian peran tim
- Sesi tanya jawab

## 4. RINGKASAN ISI RAPAT

### 4.1 Sesi Perkenalan
Rapat dibuka oleh mentor, Muhammad Ridha, dengan sesi perkenalan diri dari setiap peserta. Berikut highlight perkenalan masing-masing:

**Muhammad Mufid Taqiyuddin (Mufid)**
- Mahasiswa Sistem Informasi, Fakultas Rekayasa Industri (FRI), angkatan 2023
- Mengikuti internship di Yumik dengan skema konversi SKS
- Memiliki pengalaman dalam pengembangan machine learning dan implementasi di aplikasi
- Belum pernah menjabat sebagai Project Manager sebelumnya

**Muhammad Raihan Thaffan Hidayat (Raihan)**
- Mahasiswa Teknik Telekomunikasi, angkatan 2022, sedang dalam tahap Tugas Akhir
- Memiliki background yang cukup general: jaringan, transmisi, AI, dan sedikit web
- Pernah membuat project web menggunakan framework (RGGS, RTSS) saat mata kuliah
- Mengikuti internship di Yumik dengan skema konversi SKS

**Athila Ramdani Saputra (Attila)**
- Mahasiswa semester 6, mengikuti magang di Yumik
- Terdaftar sebagai Back-End Developer, namun pada akhir sesi ditunjuk sebagai Project Manager
- Mentor sempat mengomentari kemiripan wajah Attila dengan Febrianu Farerah, alumni FRI yang juga merupakan Mahasiswa Berprestasi (Mapress) Telkom University

### 4.2 Pemaparan Proyek: Menthealth
Mentor memaparkan gambaran besar proyek yang akan dikerjakan selama masa internship. Proyek ini bertujuan untuk:
- Mengklasifikasikan kondisi kesehatan mental seseorang berdasarkan input audio
- Tiga kelas target: Depresi, Kecemasan (Anxiety), dan Stress
- Pendekatan: Audio-only feature (bukan teks atau video)
- Metode: Perbandingan Machine Learning (ML) vs Deep Learning (DL)
- Dilengkapi dengan Explainable AI (XAI) untuk menjelaskan prediksi model
- Output akhir: Web aplikasi yang dapat diakses pengguna untuk menganalisis audio

### 4.3 Dataset
Mentor menyampaikan ketentuan terkait dataset sebagai berikut:
- Tim tidak perlu mengumpulkan data sendiri (tidak perlu crawling mandiri)
- Gunakan dataset publik yang sudah tersedia — mentor memberikan 2 referensi dataset sebagai titik awal
- Tim diperbolehkan menambahkan dataset lain selain yang diberikan mentor
- Jika mayoritas dataset berbahasa Indonesia, proyek dikerjakan dengan data bahasa Indonesia; jika berbahasa Inggris, gunakan bahasa Inggris
- Pencarian dan evaluasi dataset adalah salah satu tugas minggu pertama

### 4.4 Feature Extraction Audio
Model akan bekerja berdasarkan fitur-fitur yang diekstrak dari audio. Beberapa metode yang disebutkan mentor:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Pitch (nada dasar suara)
- Energy (kekuatan/amplitudo sinyal audio)
- Metode lain yang relevan (tim bebas mengeksplorasi)
- Seluruh pemrosesan harus menggunakan Python, direkomendasikan menggunakan Google Colaboratory

### 4.5 Pendekatan ML & DL
Proyek ini menggunakan pendekatan komparatif antara model Machine Learning klasik dan model Deep Learning:

**Machine Learning (ML):**
- Support Vector Machine (SVM)
- Random Forest
- AdaBoost
- Gradient Boosting
- Logistic Regression

**Deep Learning (DL):**
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional LSTM)
- Wav2Vec
- CNN-LSTM
- iBREAD
- Model lain yang relevan (tim dipersilakan mengeksplorasi)

Evaluasi model menggunakan metrik: Akurasi, Presisi, Recall, dan F1 Score. Semua hasil akan ditampilkan dalam bentuk tabel dan grafik perbandingan di website.

### 4.6 Explainable AI (XAI)
Fitur XAI digunakan untuk menjelaskan bagian mana dari audio yang paling berpengaruh terhadap prediksi model. Mentor menyebutkan:
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Tim diperbolehkan menggunakan metode XAI lain yang lebih relevan jika ada
- Output XAI divisualisasikan seperti grafik tensi/sinyal, menunjukkan bagian audio yang "ditekankan" oleh model

### 4.7 Arsitektur Website & Deployment
Mentor menjelaskan bahwa proyek ini wajib memiliki antarmuka web yang dapat digunakan pengguna. Beberapa poin penting:
- Frontend: Bebas menggunakan React, Vue, atau framework apapun
- Backend: Mengintegrasikan model ML/DL yang telah dibangun melalui API
- Tidak perlu fitur berlebihan (tidak perlu login, register, CRUD kompleks, autentikasi, dll.)
- Fokus pada satu landing page yang fungsional dengan fitur inti
- Halaman website maksimal 2-3 halaman

Fitur inti yang WAJIB ada di website:
- Upload file audio dari pengguna
- Tampilan hasil prediksi: klasifikasi (Depresi / Kecemasan / Stress) + Confidence Score
- Visualisasi XAI: bagian audio yang dominan/berpengaruh terhadap prediksi
- Tabel/grafik perbandingan performa model ML vs DL

## 5. OUTPUT YANG DIHARAPKAN

| No | Output | Keterangan |
| --- | --- | --- |
| 1 | Klasifikasi Kondisi Mental | Depresi, Kecemasan, Stress |
| 2 | Confidence Score | Seberapa yakin model terhadap prediksi |
| 3 | Explainability (XAI) | Menampilkan fitur audio dominan (LIME / SHAP) |
| 4 | Model Comparison | Tabel & grafik perbandingan ML vs DL (Akurasi, Presisi, Recall, F1) |

## 6. TIMELINE PENGERJAAN (± 2 Bulan)

| Minggu | Fase | Aktivitas |
| --- | --- | --- |
| 0 - 1 | Data Acquisition | Pencarian & pengunduhan dataset publik audio kesehatan mental |
| 2 | Data Preprocessing | Pembersihan data, normalisasi audio, labeling |
| 3 | Feature Extraction | Ekstraksi fitur audio: MFCC, Pitch, Energi, dll. |
| 4 | Training ML | SVM, Random Forest, AdaBoost, Gradient Boosting, Logistic Regression |
| 5 | Training DL | LSTM, BiLSTM, Wav2Vec, CNN-LSTM, iBREAD, dll. |
| 6 | Setup Front-End & Backend | Pembuatan antarmuka website & integrasi API model |
| 7 | Integrasi & XAI | Integrasi model ke website + implementasi SHAP/LIME |
| 8 | Testing & Finalisasi | Pengujian end-to-end, perbaikan bug, finalisasi aplikasi |

📌 **Catatan**: Mentor akan memperbaiki dan merinci timeline per minggu setelah pertemuan ini. Progres dilaporkan setiap hari Sabtu melalui Google Classroom.

## 7. PEMBAGIAN TIM & PERAN
Berikut adalah pembagian peran yang disepakati dalam pertemuan ini:

| Anggota | Peran | Tanggung Jawab Utama |
| --- | --- | --- |
| Athila Ramdani Saputra | Back-End Dev + PM | Koordinasi tim, pengembangan API backend, integrasi model ML/DL ke server |
| Muhammad Mufid Taqiyuddin | UI/UX Designer | Desain tampilan website (landing page, color grading, layout upload audio & visualisasi hasil) |
| Muhammad Raihan Thaffan Hidayat | Front-End Developer | Implementasi antarmuka ke kode, koneksi frontend ke API backend |

**Catatan Penentuan Project Manager:**
Awalnya tidak ada yang langsung bersedia menjadi Project Manager. Mentor mendorong peserta untuk mengajukan diri secara sukarela dengan menekankan pentingnya pengalaman memimpin proyek untuk perkembangan karir. Akhirnya Athila Ramdani Saputra bersedia menjadi Project Manager. Mentor berpesan agar Mufid dan Raihan sepenuhnya mendukung dan menghormati arahan PM.

## 8. KESEPAKATAN & KEPUTUSAN RAPAT
- Tim akan segera mencari dan mengevaluasi dataset publik audio kesehatan mental sebagai prioritas utama minggu ke-1
- Seluruh kode Python dikerjakan di Google Colaboratory
- Athila Ramdani Saputra ditetapkan sebagai Project Manager; Mufid dan Raihan berperan sebagai anggota aktif yang mendukung PM
- Website fokus pada satu landing page fungsional, tidak perlu autentikasi atau fitur CRUD kompleks
- Progres mingguan wajib di-upload ke Google Classroom setiap hari Sabtu
- Pertemuan berikutnya bersifat presentasi progres, durasi sekitar 30 menit
- Mentor akan membagikan PPT materi onboarding dan merinci timeline per minggu setelah sesi ini

## 9. TINDAK LANJUT (ACTION ITEMS)

| No | Tugas | Penanggung Jawab | Tenggat |
| --- | --- | --- | --- |
| 1 | Join Google Classroom yang akan dibuat mentor | Semua Peserta | Minggu ini |
| 2 | Mencari dan mengevaluasi dataset publik audio kesehatan mental | Semua / Athila (PM) | Minggu ke-1 |
| 3 | Membuat rencana kerja mingguan berdasarkan timeline dari mentor | Athila (PM) | Minggu ke-1 |
| 4 | Mentor membagikan PPT materi dan timeline rinci per minggu | Muhammad Ridha | Setelah sesi |
| 5 | Upload progres pertama ke Google Classroom | Semua Peserta | Sabtu ini |

## 10. CATATAN TAMBAHAN
Beberapa hal menarik yang dicatat selama pertemuan:
- Mas Ridha (mentor) menyampaikan bahwa beliau merupakan alumni Telkom dan pernah bekerja di Telkom selama 2 tahun di direktorat desain
- Mas Rayya, yang melakukan onboarding intern sebelumnya, ternyata merupakan mantan mahasiswa magang yang pernah dibimbing langsung oleh Mas Ridha
- Mentor sempat mencandai kemiripan wajah Attila dengan Febrianu Farerah (Mapress Telkom) dan meminta tim untuk membandingkannya — momen yang cukup mencairkan suasana
- Pertemuan ini dinyatakan sebagai 'Minggu ke-0' sehingga tugas minggu ke-1 sudah dimulai sejak minggu ini
- Seluruh pertemuan bersifat singkat dan efisien — estimasi durasi hanya 30 menit per sesi ke depannya


---
Pertemuan ditutup dengan salam dan ucapan terima kasih dari seluruh peserta.
*Assalamu'alaikum Warahmatullahi Wabarakatuh*

<br>

| Mengetahui | Dibuat oleh |
| :--- | :--- |
| **Mentor Proyek Menthealth** | **Project Manager** |
| Muhammad Ridha | Athila Ramdani Saputra |
