# %% [markdown]
# # Part 1 — Dataset Overview: MODMA
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (MODMA)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# Notebook ini bertugas untuk:
# 1. Scan seluruh folder partisipan di dataset MODMA
# 2. Memeriksa kelengkapan file setiap partisipan (jumlah audio .wav)
# 3. Membaca metadata dari excel untuk mendapatkan skor PHQ-9
# 4. Distribusi label: Stress | Kecemasan | Depresi
# 5. Menghasilkan `modma_metadata.csv` sebagai tabel inventori utama
#
# **Strategi Labeling 3 Kelas (PHQ-9 Severity Proxy)**:
# - PHQ-9  0-4  → Kelas 0: Stress    (gejala minimal)
# - PHQ-9  5-14 → Kelas 1: Kecemasan (gejala ringan-sedang)
# - PHQ-9 >= 15 → Kelas 2: Depresi   (gejala berat, MDD klinis)

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi Path

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR     = os.path.dirname(os.path.dirname(current_dir))  # menthealth-ml/
MODMA_DIR    = os.path.join(BASE_DIR, 'dataset', 'raw', 'MODMA')
PROCESSED_DIR= os.path.join(BASE_DIR, 'dataset', 'processed')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'modma')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"MODMA DIR    : {MODMA_DIR}")
print(f"PROCESSED DIR: {PROCESSED_DIR}")

# %% [markdown]
# ## 1.1 — Baca Metadata dari Excel

# %%
excel_path = os.path.join(MODMA_DIR, 'subjects_information_audio_lanzhou_2015.xlsx')
df_meta = pd.read_excel(excel_path)
df_meta['folder_name'] = df_meta['subject id'].apply(lambda x: f"0{x}")

print(f"Total partisipan di Excel: {len(df_meta)}")
# Mengganti spasi di nama kolom dengan underscore untuk kemudahan
df_meta.columns = [c.strip().replace(' ', '_').lower() for c in df_meta.columns]
print("Kolom tersedia:", list(df_meta.columns))

# %% [markdown]
# ## 1.2 — Scan Folder Partisipan

# %%
records = []
for idx, row in df_meta.iterrows():
    folder_name = row['folder_name']
    folder_path = os.path.join(MODMA_DIR, folder_name)
    
    has_folder = os.path.isdir(folder_path)
    wav_count = 0
    
    if has_folder:
        wav_count = len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
        
    records.append({
        'participant_id': row['subject_id'],
        'folder_path'   : folder_path,
        'has_folder'    : has_folder,
        'wav_count'     : wav_count,
        'phq9_score'    : row['phq-9'],
        'gad7_score'    : row['gad-7'],
        'type'          : row['type']
    })

df_scan = pd.DataFrame(records)

print(df_scan[['participant_id', 'has_folder', 'wav_count', 'phq9_score', 'type']].head(10).to_string(index=False))
print(f"\nPartisipan dengan folder ditemukan: {df_scan['has_folder'].sum()} / {len(df_scan)}")
print(f"Rata-rata file audio per partisipan: {df_scan['wav_count'].mean():.2f}")


# %% [markdown]
# ## 1.3 — Labeling 3 Kelas Berbasis PHQ-9 Severity

# %%
CLASS_NAMES = {0: 'Stress', 1: 'Kecemasan', 2: 'Depresi'}

def phq9_to_3class(score):
    """Konversi skor PHQ-9 ke label 3 kelas: Stress | Kecemasan | Depresi."""
    if score is None or pd.isna(score):
        return None
    score = int(score)
    if score <= 4:
        return 0   # Stress (minimal symptoms)
    elif score <= 14:
        return 1   # Kecemasan (mild-moderate)
    else:
        return 2   # Depresi (severe)

def phq9_severity(score):
    if score is None or pd.isna(score):
        return 'Unknown'
    score = int(score)
    if score <= 4:
        return 'Stress (0-4)'
    elif score <= 14:
        return 'Kecemasan (5-14)'
    else:
        return 'Depresi (15+)'

df_scan['label_3kelas'] = df_scan['phq9_score'].apply(phq9_to_3class)
df_scan['severity']     = df_scan['phq9_score'].apply(phq9_severity)

df_labeled = df_scan[df_scan['phq9_score'].notna()].copy()
print("\n=== Statistik PHQ-9 ===")
print(df_labeled['phq9_score'].describe().round(2))
print("\n=== Distribusi 3 Kelas ===")
for k, name in CLASS_NAMES.items():
    n = (df_labeled['label_3kelas'] == k).sum()
    print(f"  Kelas {k} ({name:20s}): {n} partisipan ({n/len(df_labeled)*100:.1f}%)")

# %% [markdown]
# ## 1.4 — Visualisasi Distribusi Dataset MODMA

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Distribusi Dataset MODMA — Klasifikasi 3 Kelas', fontsize=14, fontweight='bold')

class_order  = [0, 1, 2]
class_labels = [CLASS_NAMES[k] for k in class_order]
class_counts = [int((df_labeled['label_3kelas'] == k).sum()) for k in class_order]

bars1 = axes[0].bar(class_labels, class_counts,
    color=['#3498db', '#f39c12', '#e74c3c'], edgecolor='black', linewidth=0.8)
axes[0].set_title('Distribusi Label 3 Kelas')
axes[0].set_ylabel('Jumlah Partisipan')
for bar, val in zip(bars1, class_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha='center', fontweight='bold')

axes[1].pie(class_counts, labels=class_labels, autopct='%1.1f%%',
            colors=['#3498db', '#f39c12', '#e74c3c'], startangle=90,
            textprops={'fontsize': 9})
axes[1].set_title('Proporsi Kelas (3-Class)')

axes[2].hist(df_labeled['phq9_score'].dropna(), bins=10, color='#3498db',
             edgecolor='black', alpha=0.85)
axes[2].axvline(x=5,  color='#f39c12', linestyle='--', linewidth=1.5, label='Threshold Kec/Stress = 5')
axes[2].axvline(x=15, color='#e74c3c', linestyle='--', linewidth=1.5, label='Threshold Depresi = 15')
axes[2].set_title('Histogram Skor PHQ-9 + Threshold 3 Kelas')
axes[2].set_xlabel('Skor PHQ-9')
axes[2].set_ylabel('Frekuensi')
axes[2].legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p1_distribusi_dataset_modma.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi tersimpan.")

# %% [markdown]
# ## 1.5 — Simpan Metadata ke CSV

# %%
OUTPUT_META = os.path.join(PROCESSED_DIR, 'modma_metadata.csv')
df_scan.to_csv(OUTPUT_META, index=False)

print(f"Metadata disimpan: {OUTPUT_META}")
print(f"\nRingkasan Akhir:")
print(f"  Total partisipan       : {len(df_scan)}")
print(f"  File audio rata-rata   : {df_scan['wav_count'].mean():.1f}")
for k, name in CLASS_NAMES.items():
    n = int((df_labeled['label_3kelas'] == k).sum())
    print(f"  Kelas {k} ({name}): {n}")
