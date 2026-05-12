import matplotlib
matplotlib.use('Agg')
import subprocess, sys, os

env = os.environ.copy()
env['MPLBACKEND'] = 'Agg'

parts = [
    ('1', 'Part1_Dataset_Overview'),
    ('2', 'Part2_Preprocessing'),
    ('3', 'Part3_Feature_Extraction'),
    ('4', 'Part4_Dataset_Building'),
    ('5', 'Part5_Split_Data'),
    ('6', 'Part6_Training_Model'),
    ('7', 'Part7_XAI'),
]

results = {}
for num, name in parts:
    path = f'experiments/daic/{name}.py'
    print(f'Running Part {num}: {name}...', flush=True)
    try:
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=600, env=env
        )
        last_lines = [l for l in r.stdout.strip().split('\n') if l.strip()][-3:]
        last_out = last_lines[-1] if last_lines else ''
        if r.returncode == 0:
            results[name] = ('OK', last_out)
            print(f'  OK | {last_out}')
        else:
            err_lines = [l for l in r.stderr.strip().split('\n') if l.strip()][-3:]
            err = err_lines[-1] if err_lines else 'unknown'
            results[name] = ('ERROR', err)
            print(f'  ERROR | {err}')
            # Print lebih detail untuk debug
            for l in err_lines:
                print(f'    {l}')
    except subprocess.TimeoutExpired:
        results[name] = ('TIMEOUT', 'Melebihi 600 detik')
        print(f'  TIMEOUT')
    except Exception as e:
        results[name] = ('EXCEPTION', str(e))
        print(f'  EXCEPTION: {e}')

print()
print('='*60)
print('RANGKUMAN HASIL TEST PIPELINE DAIC-WOZ')
print('='*60)
for name, (status, msg) in results.items():
    icon = 'OK' if status == 'OK' else 'FAIL'
    print(f'[{icon}] {name}: {msg[:80]}')
