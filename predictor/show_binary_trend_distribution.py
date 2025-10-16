import os
import glob
import pandas as pd
import json

root = os.path.dirname(os.path.abspath(__file__))
core_data = os.path.join(os.path.dirname(root), 'core', 'data')
patterns = [os.path.join(core_data, 'posts_*with_trends*.csv'), os.path.join(core_data, 'posts_*_with_trends*.csv')]
files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(set(files))

if not files:
    print('No se encontraron archivos with_trends en core/data')
    raise SystemExit(1)

print(f'Archivos encontrados: {len(files)}')

rows = 0
zeros = 0
ones = 0
missing = 0
per_file_stats = []

for f in files:
    df = pd.read_csv(f)
    rows += len(df)
    if 'trend_similarity' not in df.columns:
        missing += len(df)
        per_file_stats.append((os.path.basename(f), len(df), 0, 0, True))
        continue
    ts = pd.to_numeric(df['trend_similarity'], errors='coerce').fillna(0)
    bin_flag = (ts > 0).astype(int)
    z = int((bin_flag == 0).sum())
    o = int((bin_flag == 1).sum())
    zeros += z
    ones += o
    per_file_stats.append((os.path.basename(f), len(df), z, o, False))

report = {
    'files_count': len(files),
    'rows': rows,
    'missing_trend_similarity_total': missing,
    'zeros_total': zeros,
    'ones_total': ones,
    'percent_ones': float(ones / rows * 100) if rows>0 else 0.0,
    'per_file': [
        {'file': p[0], 'rows': p[1], 'zeros': p[2], 'ones': p[3], 'missing_col': p[4]} for p in per_file_stats
    ]
}

print(json.dumps(report, indent=2, ensure_ascii=False))

# Optionally save a CSV with the binarized column for inspection
out = os.path.join(root, 'trend_similarity_binary_summary.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('\nResumen guardado en', out)
