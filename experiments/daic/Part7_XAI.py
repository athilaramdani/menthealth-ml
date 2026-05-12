# %% [markdown]
# # Part 7 — XAI (Explainable AI): DAIC-WOZ
# **Kelas**: 0=Stress | 1=Kecemasan | 2=Depresi
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# - SHAP multiclass: beeswarm & waterfall per kelas (RF & XGBoost)
# - LIME multiclass: local explanation per instance (SVM & LR)
# - Visualisasi: Prediksi vs Aktual, Top Feature Importance

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
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

# Install lime jika belum ada
try:
    from lime.lime_tabular import LimeTabularExplainer
    print("lime tersedia.")
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'lime'], check=True)
    from lime.lime_tabular import LimeTabularExplainer
    print("lime berhasil diinstall.")

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR       = os.path.dirname(os.path.dirname(current_dir))
PROCESSED_DIR  = os.path.join(BASE_DIR, 'dataset', 'processed')
MODELS_DIR     = os.path.join(PROCESSED_DIR, 'models')
OUTPUT_DIR     = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')

TRAIN_PATH     = os.path.join(PROCESSED_DIR, 'daic_train.csv')
TEST_PATH      = os.path.join(PROCESSED_DIR, 'daic_test.csv')
FEAT_LIST_PATH = os.path.join(PROCESSED_DIR, 'daic_feature_list.txt')

CLASS_NAMES    = {0: 'Stress', 1: 'Kecemasan', 2: 'Depresi'}
CLASS_COLORS   = {0: '#3498db', 1: '#f39c12', 2: '#e74c3c'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ## 7.1 — Load Data & Model

# %%
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

with open(FEAT_LIST_PATH, 'r') as f:
    FEAT_COLS = [line.strip() for line in f if line.strip()]
FEAT_COLS = [c for c in FEAT_COLS if c in df_train.columns]

X_train = df_train[FEAT_COLS].values
y_train = df_train['label_3kelas'].values
X_test  = df_test[FEAT_COLS].values
y_test  = df_test['label_3kelas'].values

def load_model(name):
    safe_name  = name.replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

model_rf  = load_model('Random_Forest')
model_xgb = load_model('XGBoost')
model_svm = load_model('SVM_RBF')
model_lr  = load_model('Logistic_Regression')

loaded = {n: m for n, m in [('Random Forest', model_rf), ('XGBoost', model_xgb),
                              ('SVM (RBF)', model_svm), ('Logistic Regression', model_lr)]
          if m is not None}

print(f"Model dimuat: {list(loaded.keys())}")
print(f"Test: {len(X_test)} sampel | {len(FEAT_COLS)} fitur")
print(f"Distribusi kelas test:")
for k, name in CLASS_NAMES.items():
    print(f"  Kelas {k} ({name}): {(y_test==k).sum()}")

# %% [markdown]
# ## 7.2 — SHAP: Random Forest (Multiclass)
# SHAP untuk multiclass menghasilkan satu set SHAP values per kelas.
# Kita visualisasikan untuk setiap kelas: Stress, Kecemasan, Depresi.

# %%
if model_rf is not None:
    print("Menghitung SHAP values (Random Forest, multiclass)...")
    X_shap = X_test[:min(80, len(X_test))]
    y_shap = y_test[:min(80, len(y_test))]

    explainer_rf    = shap.TreeExplainer(model_rf)
    shap_values_rf  = explainer_rf.shap_values(X_shap)
    # shap_values_rf adalah list of arrays: [kelas0, kelas1, kelas2]

    # --- Beeswarm per kelas ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SHAP Beeswarm per Kelas — Random Forest\n(Stress | Kecemasan | Depresi)',
                 fontsize=12, fontweight='bold')

    for ci, (k, name) in enumerate(CLASS_NAMES.items()):
        sv = shap_values_rf[k] if isinstance(shap_values_rf, list) else shap_values_rf[:, :, k]
        plt.sca(axes[ci])
        shap.summary_plot(sv, X_shap, feature_names=FEAT_COLS,
                          max_display=12, show=False, plot_type='dot')
        axes[ci].set_title(f'Kelas {k}: {name}', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'p7_shap_rf_beeswarm_3class.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("SHAP Beeswarm 3-kelas tersimpan.")

    # --- Global Feature Importance (mean |SHAP| across all classes) ---
    if isinstance(shap_values_rf, list):
        mean_abs = np.mean([np.abs(shap_values_rf[k]).mean(axis=0) for k in range(3)], axis=0)
    else:
        mean_abs = np.abs(shap_values_rf).mean(axis=(0, 2))

    top_n   = 15
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_f   = [FEAT_COLS[i] for i in top_idx]
    top_v   = [mean_abs[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_f[::-1], top_v[::-1], color='#8e44ad', edgecolor='black', linewidth=0.6, alpha=0.85)
    ax.set_xlabel('Mean |SHAP Value| (rata-rata 3 kelas)')
    ax.set_title(f'Top {top_n} Fitur Terpenting — Random Forest (SHAP)\nDATAC-WOZ | 3 Kelas: Stress | Kecemasan | Depresi',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'p7_shap_rf_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("SHAP Global Importance tersimpan.")
else:
    print("Model RF tidak ditemukan.")

# %% [markdown]
# ## 7.3 — SHAP: Waterfall Plot Lokal (1 sampel per kelas)
# Menjelaskan mengapa model memprediksi kelas tertentu untuk 1 individu.

# %%
if model_rf is not None and isinstance(shap_values_rf, list):
    for k, name in CLASS_NAMES.items():
        idx_list = np.where(y_shap == k)[0]
        if len(idx_list) == 0:
            print(f"Tidak ada sampel kelas {k} ({name}) di subset test.")
            continue

        sample_idx  = idx_list[0]
        pred_label  = model_rf.predict([X_shap[sample_idx]])[0]
        sv_k        = shap_values_rf[k]
        exp_val     = explainer_rf.expected_value[k] if isinstance(explainer_rf.expected_value, list) else explainer_rf.expected_value

        shap_expl = shap.Explanation(
            values        = sv_k[sample_idx],
            base_values   = exp_val,
            data          = X_shap[sample_idx],
            feature_names = FEAT_COLS,
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(shap_expl, max_display=12, show=False)
        plt.title(f'SHAP Waterfall — Kelas {k}: {name}\n'
                  f'Aktual={CLASS_NAMES.get(y_shap[sample_idx],"?")} | Prediksi={CLASS_NAMES.get(pred_label,"?")}',
                  fontweight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'p7_shap_waterfall_kelas{k}_{name.lower()}.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(f"SHAP Waterfall kelas {k} ({name}) tersimpan.")

# %% [markdown]
# ## 7.4 — LIME: SVM & Logistic Regression (Multiclass)
# LIME mendukung multiclass secara native via `top_labels=3`.

# %%
lime_explainer = LimeTabularExplainer(
    training_data = X_train,
    feature_names = FEAT_COLS,
    class_names   = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES.keys())],
    mode          = 'classification',
    random_state  = 42,
)

def run_lime_3class(model, model_name, X_sample, actual_label, n_features=12):
    """Jalankan LIME multiclass dan simpan plot per kelas prediksi tertinggi."""
    try:
        lime_exp = lime_explainer.explain_instance(
            data_row  = X_sample,
            predict_fn= model.predict_proba,
            num_features = n_features,
            num_samples  = 500,
            top_labels   = 3,
        )
        pred_label = model.predict([X_sample])[0]
        pred_proba = model.predict_proba([X_sample])[0]

        # Tampilkan untuk kelas yang diprediksi
        exp_list   = lime_exp.as_list(label=pred_label)
        feat_names = [e[0] for e in exp_list]
        feat_vals  = [e[1] for e in exp_list]

        sorted_pairs = sorted(zip(feat_vals, feat_names))
        s_vals, s_names = zip(*sorted_pairs) if sorted_pairs else ([], [])

        colors = ['#e74c3c' if v > 0 else '#3498db' for v in s_vals]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(s_names), list(s_vals), color=colors, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=1.2)
        ax.set_xlabel(f'Kontribusi terhadap Kelas: {CLASS_NAMES.get(pred_label,"?")}')
        ax.set_title(
            f'LIME — {model_name}\n'
            f'Aktual={CLASS_NAMES.get(actual_label,"?")} | '
            f'Prediksi={CLASS_NAMES.get(pred_label,"?")} | '
            f'P={pred_proba[pred_label]:.3f}',
            fontweight='bold', fontsize=10
        )
        plt.tight_layout()
        safe_m = model_name.lower().replace(' ','_').replace('(','').replace(')','')
        safe_l = CLASS_NAMES.get(actual_label,'').lower()
        path   = os.path.join(OUTPUT_DIR, f'p7_lime_{safe_m}_{safe_l}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"LIME [{model_name}] kelas aktual {CLASS_NAMES.get(actual_label,'')} tersimpan.")
    except Exception as e:
        print(f"  LIME [{model_name}] gagal: {e}")

# %% [markdown]
# ## 7.5 — Jalankan LIME untuk SVM dan LR

# %%
for lm_name, lm_model in [('SVM (RBF)', model_svm), ('Logistic Regression', model_lr)]:
    if lm_model is None:
        print(f"Model {lm_name} tidak tersedia, skip.")
        continue
    print(f"\n=== LIME: {lm_name} ===")
    # 1 sampel per kelas
    for k in CLASS_NAMES.keys():
        idx_list = np.where(y_test == k)[0]
        if len(idx_list) == 0:
            continue
        run_lime_3class(lm_model, lm_name, X_test[idx_list[0]], y_test[idx_list[0]])

# %% [markdown]
# ## 7.6 — Prediksi vs Aktual (Semua Model, 3 Kelas)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prediksi vs Aktual — DAIC-WOZ (Test Set)\n0=Stress | 1=Kecemasan | 2=Depresi',
             fontsize=12, fontweight='bold')

from sklearn.metrics import f1_score as f1_fn

for ax, (model_name, model) in zip(axes.flatten(), loaded.items()):
    y_pred   = model.predict(X_test)
    correct  = (y_pred == y_test)
    np.random.seed(42)
    jitter   = np.random.uniform(-0.15, 0.15, len(y_test))

    for k, name in CLASS_NAMES.items():
        mask_c = correct & (y_test == k)
        mask_w = ~correct & (y_test == k)
        ax.scatter(y_test[mask_c] + jitter[mask_c], y_pred[mask_c],
                   c=CLASS_COLORS[k], alpha=0.6, s=50,
                   label=f'{name} ✓' if k == 0 else None,
                   edgecolors='white', linewidths=0.3)
        ax.scatter(y_test[mask_w] + jitter[mask_w], y_pred[mask_w],
                   c=CLASS_COLORS[k], alpha=0.6, s=50, marker='x',
                   edgecolors=CLASS_COLORS[k])

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Stress', 'Kecemasan', 'Depresi'], fontsize=8)
    ax.set_yticklabels(['Stress', 'Kecemasan', 'Depresi'], fontsize=8)
    ax.set_xlabel('Aktual')
    ax.set_ylabel('Prediksi')
    macro_f1 = f1_fn(y_test, y_pred, average='macro', zero_division=0)
    acc      = (y_pred == y_test).mean()
    ax.set_title(f'{model_name}\nMacro F1={macro_f1:.3f} | Acc={acc:.3f}',
                 fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Legend simpel
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Benar'),
        Line2D([0],[0], marker='x', color='gray', markersize=8, label='Salah'),
    ]
    ax.legend(handles=legend_elems, fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p7_pred_vs_actual_3class.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi prediksi vs aktual tersimpan.")

# %% [markdown]
# ## 7.7 — Ringkasan Top Fitur (Semua Kelas)

# %%
if model_rf is not None and isinstance(shap_values_rf, list):
    print("\n" + "="*60)
    print("TOP 10 FITUR PER KELAS — Random Forest (SHAP)")
    print("="*60)
    for k, name in CLASS_NAMES.items():
        sv_k      = shap_values_rf[k]
        mean_abs_k= np.abs(sv_k).mean(axis=0)
        top_idx_k = np.argsort(mean_abs_k)[::-1][:10]
        print(f"\nKelas {k}: {name}")
        for rank, idx in enumerate(top_idx_k, 1):
            print(f"  {rank:2d}. {FEAT_COLS[idx]:<35} |SHAP|={mean_abs_k[idx]:.5f}")

print("\n✓ Analisis XAI selesai.")
print("  Output:", OUTPUT_DIR)
