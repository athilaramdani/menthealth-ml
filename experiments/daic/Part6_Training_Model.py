# %% [markdown]
# # Part 6 — Training Model ML: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# Notebook ini bertugas untuk:
# 1. Load data train/val/test dari Part 5 (sudah di-scale)
# 2. Train 4 model ML: Logistic Regression, SVM, Random Forest, XGBoost
# 3. Cross-validation dengan GroupKFold (5-fold, anti-leakage)
# 4. Hyperparameter tuning dengan GridSearchCV
# 5. Evaluasi MULTICLASS: Macro F1, Accuracy, Per-Class Report, Confusion Matrix 3x3
# 6. Simpan model terbaik & tabel perbandingan seluruh model
#
# **Kelas Target**: 0=Stress | 1=Kecemasan | 2=Depresi (via PHQ-8 proxy)
# **Referensi**: Yadav et al. (2023), Danylenko & Unold (2025)

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
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             mean_absolute_error, r2_score)
import xgboost as xgb
import pickle
import json
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

BASE_DIR      = os.path.dirname(os.path.dirname(current_dir))
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
MODELS_DIR    = os.path.join(PROCESSED_DIR, 'models')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')

TRAIN_PATH    = os.path.join(PROCESSED_DIR, 'daic_train.csv')
VAL_PATH      = os.path.join(PROCESSED_DIR, 'daic_val.csv')
TEST_PATH     = os.path.join(PROCESSED_DIR, 'daic_test.csv')
FEAT_LIST_PATH= os.path.join(PROCESSED_DIR, 'daic_feature_list.txt')
SPLIT_INFO    = os.path.join(PROCESSED_DIR, 'daic_split_info.json')

os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42
print(f"Models dir: {MODELS_DIR}")

# %% [markdown]
# ## 6.1 — Load Data

# %%
df_train = pd.read_csv(TRAIN_PATH)
df_val   = pd.read_csv(VAL_PATH)
df_test  = pd.read_csv(TEST_PATH)

with open(FEAT_LIST_PATH, 'r') as f:
    FEAT_COLS = [line.strip() for line in f if line.strip()]
FEAT_COLS = [c for c in FEAT_COLS if c in df_train.columns]

META_COLS = ['participant_id', 'phq8_score', 'label_3kelas', 'severity']

X_train = df_train[FEAT_COLS].values
y_train = df_train['label_3kelas'].values   # 0=Stress, 1=Kecemasan, 2=Depresi
groups_train = df_train['participant_id'].values

X_val   = df_val[FEAT_COLS].values
y_val   = df_val['label_3kelas'].values

X_test  = df_test[FEAT_COLS].values
y_test  = df_test['label_3kelas'].values

# Gabung train+val untuk training final
X_trainval     = np.vstack([X_train, X_val])
y_trainval     = np.concatenate([y_train, y_val])
groups_trainval= np.concatenate([groups_train, df_val['participant_id'].values])

CLASS_NAMES = {0: 'Stress', 1: 'Kecemasan', 2: 'Depresi'}

print(f"Train : {X_train.shape}")
for k, name in CLASS_NAMES.items():
    print(f"  {name}: {(y_train==k).sum()}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")
print(f"Fitur : {len(FEAT_COLS)}")

# %% [markdown]
# ## 6.2 — Definisi Model & Hyperparameter Grid
# SVM multiclass menggunakan `decision_function_shape='ovr'` (One-vs-Rest).
# class_weight='balanced' untuk handle imbalance antar kelas.

# %%
MODELS = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                                    class_weight='balanced',
                                    multi_class='auto'),
        'param_grid': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
        }
    },
    'SVM (RBF)': {
        'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED,
                     class_weight='balanced',
                     decision_function_shape='ovr'),  # One-vs-Rest untuk multiclass
        'param_grid': {
            'C':     [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto'],
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_SEED,
                                        class_weight='balanced', n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth':    [None, 10, 20],
            'min_samples_split': [2, 5],
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='mlogloss',
                                    num_class=3,       # 3 kelas
                                    objective='multi:softmax',
                                    verbosity=0, n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth':    [3, 5, 7],
            'learning_rate':[0.05, 0.1],
        }
    }
}

print("Model yang akan dilatih:")
for name in MODELS.keys():
    print(f"  - {name}")

# %% [markdown]
# ## 6.3 — Fungsi Evaluasi

# %%
def evaluate_model(model, X, y_true, prefix=''):
    """Hitung semua metrik evaluasi multiclass (macro-averaged)."""
    y_pred = model.predict(X)
    return {
        f'{prefix}accuracy' : accuracy_score(y_true, y_pred),
        f'{prefix}f1_macro' : f1_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        f'{prefix}precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}recall'   : recall_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}mae'      : mean_absolute_error(y_true, y_pred),
    }

# %% [markdown]
# ## 6.4 — Training dengan GroupKFold Cross-Validation + GridSearch

# %%
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

results      = {}
best_models  = {}

print("="*65)
print(f"{'TRAINING MODEL':^65}")
print("="*65)

for model_name, model_cfg in MODELS.items():
    print(f"\n[{model_name}]")

    base_model  = model_cfg['model']
    param_grid  = model_cfg['param_grid']

    # GridSearchCV dengan GroupKFold (anti-leakage)
    gsearch = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        cv         = gkf,
        scoring    = 'f1_macro',   # Macro F1 untuk multiclass
        n_jobs     = -1,
        verbose    = 0,
        refit      = True,
    )

    # Fit dengan groups untuk mencegah leakage dalam CV
    gsearch.fit(X_train, y_train, groups=groups_train)
    best_model = gsearch.best_estimator_

    print(f"  Best params : {gsearch.best_params_}")
    print(f"  Best CV F1  : {gsearch.best_score_:.4f}")

    # Evaluasi pada val set
    val_metrics  = evaluate_model(best_model, X_val, y_val, prefix='val_')
    test_metrics = evaluate_model(best_model, X_test, y_test, prefix='test_')

    print(f"  Best CV Macro F1: {gsearch.best_score_:.4f}")
    print(f"  Val  Macro F1   : {val_metrics['val_f1_macro']:.4f} | Acc: {val_metrics['val_accuracy']:.4f}")
    print(f"  Test Macro F1   : {test_metrics['test_f1_macro']:.4f} | Acc: {test_metrics['test_accuracy']:.4f}")

    results[model_name] = {
        'best_params'  : gsearch.best_params_,
        'cv_f1'        : gsearch.best_score_,
        **val_metrics,
        **test_metrics,
    }
    best_models[model_name] = best_model

print("\n" + "="*65)
print("Training selesai!")

# %% [markdown]
# ## 6.5 — Tabel Perbandingan Model

# %%
df_results = pd.DataFrame([
    {
        'Model'           : name,
        'CV Macro F1'     : f"{r['cv_f1']:.4f}",
        'Val Accuracy'    : f"{r['val_accuracy']:.4f}",
        'Val Macro F1'    : f"{r['val_f1_macro']:.4f}",
        'Val Weighted F1' : f"{r['val_f1_weighted']:.4f}",
        'Val Precision'   : f"{r['val_precision']:.4f}",
        'Val Recall'      : f"{r['val_recall']:.4f}",
        'Test Accuracy'   : f"{r['test_accuracy']:.4f}",
        'Test Macro F1'   : f"{r['test_f1_macro']:.4f}",
        'Test Weighted F1': f"{r['test_f1_weighted']:.4f}",
        'Test Precision'  : f"{r['test_precision']:.4f}",
        'Test Recall'     : f"{r['test_recall']:.4f}",
    }
    for name, r in results.items()
])

print("\n=== PERBANDINGAN MODEL — DAIC-WOZ ===\n")
print(df_results.to_string(index=False))

# Simpan tabel
df_results.to_csv(os.path.join(PROCESSED_DIR, 'daic_model_comparison.csv'), index=False)
print("\nTabel perbandingan tersimpan.")

# %% [markdown]
# ## 6.6 — Visualisasi Perbandingan Model

# %%
metrics_to_plot = {
    'test_f1_macro'   : 'Macro F1-Score (Test)',
    'test_accuracy'   : 'Accuracy (Test)',
    'test_precision'  : 'Macro Precision (Test)',
    'test_recall'     : 'Macro Recall (Test)',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Perbandingan Model ML — DAIC-WOZ', fontsize=14, fontweight='bold')

model_names = list(results.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for ax, (metric_key, metric_label) in zip(axes.flatten(), metrics_to_plot.items()):
    values = [float(results[m][metric_key]) for m in model_names]
    bars   = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_title(metric_label, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(y=0.60, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='Min target ≥60%')   # Multiclass lebih realistis
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p6_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi perbandingan model tersimpan.")

# %% [markdown]
# ## 6.7 — Confusion Matrix 3x3 Semua Model

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Confusion Matrix 3 Kelas — DAIC-WOZ (Test Set)\n(Stress=0 | Kecemasan=1 | Depresi=2)',
             fontsize=13, fontweight='bold')

class_tick_labels = ['Stress\n(0)', 'Kecemasan\n(1)', 'Depresi\n(2)']

for ax, (model_name, model) in zip(axes.flatten(), best_models.items()):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_tick_labels, yticklabels=class_tick_labels,
                linewidths=0.5, linecolor='gray')
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    ax.set_title(f'{model_name}\n(Macro F1={f1:.3f})', fontweight='bold', fontsize=10)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p6_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix tersimpan.")

# %% [markdown]
# ## 6.8 — Pilih & Simpan Model Terbaik

# %%
# Pilih model terbaik berdasarkan Test Macro F1
best_model_name = max(results, key=lambda m: float(results[m]['test_f1_macro']))
best_model_obj  = best_models[best_model_name]
best_f1         = float(results[best_model_name]['test_f1_macro'])

print(f"\n{'='*55}")
print(f"MODEL TERBAIK  : {best_model_name}")
print(f"Test Macro F1  : {best_f1:.4f}")
print(f"Test Accuracy  : {float(results[best_model_name]['test_accuracy']):.4f}")
print(f"{'='*55}")
print(f"\nPer-Class Classification Report:")
y_pred_best = best_model_obj.predict(X_test)
print(classification_report(y_test, y_pred_best,
      target_names=['Stress (0)', 'Kecemasan (1)', 'Depresi (2)'], zero_division=0))

# Simpan semua model
for name, model in best_models.items():
    safe_name   = name.replace(' ', '_').replace('(', '').replace(')', '')
    model_path  = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Tersimpan: {model_path}")

# Simpan info model terbaik
best_info = {
    'best_model_name' : best_model_name,
    'best_params'     : results[best_model_name]['best_params'],
    'test_f1_macro'   : float(results[best_model_name]['test_f1_macro']),
    'test_accuracy'   : float(results[best_model_name]['test_accuracy']),
    'feature_count'   : len(FEAT_COLS),
}
with open(os.path.join(MODELS_DIR, 'best_model_info.json'), 'w') as f:
    json.dump(best_info, f, indent=2)

print(f"\nBest model info tersimpan: {os.path.join(MODELS_DIR, 'best_model_info.json')}")
