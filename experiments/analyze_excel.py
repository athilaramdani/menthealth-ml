import pandas as pd
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

file_path = r'd:\repositories\menthealth-ml\dataset\raw\MODMA\subjects_information_audio_lanzhou_2015.xlsx'
df = pd.read_excel(file_path)

print('='*50)
print('INFORMASI UMUM DATASET MODMA')
print('='*50)
print(f'Total Baris (Partisipan): {len(df)}')
print(f'Total Kolom: {len(df.columns)}')
print(f'Nama Kolom: {list(df.columns)}')

print('\n' + '='*50)
print('DISTRIBUSI KELAS ASLI (Kolom: type)')
print('='*50)
print(df['type'].value_counts().to_string())

print('\n' + '='*50)
print('STATISTIK SKOR KLINIS (PHQ-9 & GAD-7)')
print('='*50)
print(df[['PHQ-9', 'GAD-7']].describe().to_string())

print('\n' + '='*50)
print('PENGECEKAN KEKOSONGAN DATA (Missing Values)')
print('='*50)
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.sum() > 0 else 'Tidak ada data kosong (Semua lengkap!)')

print('\n' + '='*50)
print('SIMULASI 3 KELAS (Normal, Depresi, Kecemasan)')
print('='*50)
def get_label(row):
    phq = row['PHQ-9']
    gad = row['GAD-7']
    diag = row['type']
    if diag == 'HC' and phq < 10 and gad < 10:
        return 'Normal'
    elif phq >= 10 and phq >= gad:
        return 'Depresi'
    elif gad >= 10 and gad > phq:
        return 'Kecemasan'
    else:
        return 'Depresi' if diag == 'MDD' else 'Normal'

df['Simulated_Class'] = df.apply(get_label, axis=1)
print(df['Simulated_Class'].value_counts().to_string())
