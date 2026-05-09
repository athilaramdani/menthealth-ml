# %% [markdown]
# # W1-W2 Feature Extraction (MODMA)
# Skrip untuk Data Preprocessing & Feature Extraction pada dataset MODMA.
# Fitur yang diekstrak: MFCC, Pitch, Energy, Spectral Features.
# Output: `modma_features.csv` (berisi 3 kelas: Normal, Depresi, Kecemasan)

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Konfigurasi Path

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR = os.path.dirname(current_dir)
MODMA_DIR = os.path.join(BASE_DIR, 'dataset', 'raw', 'MODMA')
EXCEL_PATH = os.path.join(MODMA_DIR, 'subjects_information_audio_lanzhou_2015.xlsx')
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

# %% [markdown]
# ## Fungsi: Membaca Label dari Excel
# Fungsi ini memecah kelas partisipan menjadi 3 kategori berdasarkan nilai PHQ-9 dan GAD-7.

# %%
def get_modma_labels(excel_path):
    print(f"Membaca label dari: {excel_path}")
    df = pd.read_excel(excel_path)
    labels = {}
    for idx, row in df.iterrows():
        pid = str(int(row['subject id'])).zfill(8)
        diag = row['type']
        phq = row['PHQ-9']
        gad = row['GAD-7']
        
        # Logika 3 Kelas (0: Normal, 1: Depresi, 2: Kecemasan)
        if diag == 'HC' and phq < 10 and gad < 10:
            label = 0
        elif phq >= 10 and phq >= gad:
            label = 1
        elif gad >= 10 and gad > phq:
            label = 2
        else:
            # Fallback
            label = 1 if diag == 'MDD' else 0
            
        labels[pid] = label
    return labels

# %% [markdown]
# ## Fungsi: Ekstraksi Fitur Akustik
# Mengekstrak MFCC, Pitch (Piptrack), Energy (RMS), dan Spectral features menggunakan `librosa`.

# %%
def extract_features_from_audio(y, sr):
    features = {}
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
    # Pitch (F0) dengan piptrack (cepat)
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # Ambil pitch dominan per frame
        pitch_vals = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_vals.append(pitch)
        
        if len(pitch_vals) > 0:
            features['pitch_mean'] = np.mean(pitch_vals)
            features['pitch_std'] = np.std(pitch_vals)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
    except:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
    
    # Energy / RMS
    rms = librosa.feature.rms(y=y)[0]
    features['energy_mean'] = np.mean(rms)
    features['energy_std'] = np.std(rms)
    
    # Spectral Centroid & Bandwidth
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(cent)
    features['spectral_bandwidth_mean'] = np.mean(bw)
    
    return features

# %% [markdown]
# ## Eksekusi Utama (Main Processing)
# Melakukan iterasi ke seluruh partisipan MODMA, menggabungkan audio, dan menyimpan `.csv`.

# %%
def main():
    print("="*50)
    print("Memulai Ekstraksi Fitur MODMA")
    print("="*50)
    
    modma_labels = get_modma_labels(EXCEL_PATH)
    counts = pd.Series(modma_labels.values()).value_counts().to_dict()
    print(f"Sebaran Kelas: {counts} (0: Normal, 1: Depresi, 2: Kecemasan)\n")
    
    dataset = []
    failed = []
    
    for folder in sorted(os.listdir(MODMA_DIR)):
        folder_path = os.path.join(MODMA_DIR, folder)
        if not os.path.isdir(folder_path) or folder not in modma_labels:
            continue
            
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        if not wav_files:
            continue
            
        print(f"Memproses partisipan: {folder} (Kelas: {modma_labels[folder]})...", flush=True)
        
        y_combined = []
        sr = 16000 # Downsample ke 16kHz untuk standar deteksi suara
        for wf in sorted(wav_files):
            try:
                # Load dengan sr=16000 agar lebih ringan dan standar
                y, _ = librosa.load(os.path.join(folder_path, wf), sr=sr)
                y_combined.extend(y)
            except Exception as e:
                print(f"  Error loading {wf}: {e}", flush=True)
                continue
                
        if not y_combined:
            failed.append(folder)
            continue
            
        y_combined = np.array(y_combined)
        
        try:
            features = extract_features_from_audio(y_combined, sr)
            features['participant_id'] = folder
            features['Final_Label'] = modma_labels[folder]
            dataset.append(features)
        except Exception as e:
            print(f"  Gagal ekstrak fitur partisipan {folder}: {e}", flush=True)
            failed.append(folder)

    if failed:
        print("\nPartisipan yang gagal diproses:", failed)
        
    if dataset:
        df_final = pd.DataFrame(dataset)
        cols = ['participant_id', 'Final_Label'] + [c for c in df_final.columns if c not in ['participant_id', 'Final_Label']]
        df_final = df_final[cols]
        
        output_path = os.path.join(PROCESSED_DIR, 'modma_features.csv')
        df_final.to_csv(output_path, index=False)
        print(f"\nBerhasil! {len(df_final)} partisipan tersimpan di:")
        print(output_path)
    else:
        print("\nTidak ada data yang berhasil diproses.")

if __name__ == "__main__":
    main()
