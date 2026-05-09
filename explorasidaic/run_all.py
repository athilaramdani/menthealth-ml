"""
run_all.py — Jalankan SEMUA script EDA secara berurutan.
Semua output (plot & CSV) tersimpan di folder explorasi/output/

Urutan:
  1. A_struktur_data.py   — Integritas & kelengkapan dataset
  2. B_audio_wav.py       — Eksplorasi file audio .wav
  3. C_covarep.py         — Eksplorasi fitur akustik COVAREP
  4. D_formant.py         — Eksplorasi formant frequencies
  5. E_transkrip.py       — Eksplorasi transkrip percakapan
  6. G_fitur_lanjutan.py  — Fitur MFCC, Spectral, Jitter, Shimmer

Jalankan:
  .\.venv\Scripts\activate
  python explorasi/run_all.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPTS = [
    ("A — Struktur & Integritas Data",   "explorasi.A_struktur_data"),
    ("B — Eksplorasi Audio .wav",        "explorasi.B_audio_wav"),
    ("C — Eksplorasi COVAREP",           "explorasi.C_covarep"),
    ("D — Eksplorasi Formant",           "explorasi.D_formant"),
    ("E — Eksplorasi Transkrip",         "explorasi.E_transkrip"),
    ("G — Fitur Lanjutan (MFCC/Spectral)","explorasi.G_fitur_lanjutan"),
]


def run():
    print("\n" + "▓"*60)
    print("  MENTHEALTH-ML — EDA DAIC-WOZ")
    print("  Menjalankan semua script eksplorasi...")
    print("▓"*60 + "\n")

    results = []
    for label, module_name in SCRIPTS:
        print(f"\n{'─'*60}")
        print(f"  ▶  {label}")
        print(f"{'─'*60}")
        start = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_name)
            # Tiap module sudah punya blok if __name__ == "__main__"
            # tapi kita panggil fungsinya langsung via importlib
            # Supaya lebih mudah, jalankan via subprocess
            import subprocess
            script_path = module_name.replace(".", os.sep) + ".py"
            script_full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), script_path)
            result = subprocess.run(
                [sys.executable, script_full],
                capture_output=False,
                text=True
            )
            elapsed = time.time() - start
            status  = "✅ SELESAI" if result.returncode == 0 else "❌ ERROR"
            results.append((label, status, f"{elapsed:.1f}s"))
        except Exception as e:
            elapsed = time.time() - start
            results.append((label, f"❌ ERROR: {e}", f"{elapsed:.1f}s"))

    # Ringkasan akhir
    print("\n" + "▓"*60)
    print("  RINGKASAN EKSEKUSI")
    print("▓"*60)
    for label, status, elapsed in results:
        print(f"  {status}  [{elapsed}]  {label}")
    print("\n  Output tersimpan di: explorasi/output/")
    print("▓"*60 + "\n")


if __name__ == "__main__":
    run()
