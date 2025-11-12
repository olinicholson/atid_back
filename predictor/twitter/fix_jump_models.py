import pickle
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / 'models_uala_v3_jumpboosted'
print('MODEL_DIR', MODEL_DIR)
pk_path = MODEL_DIR / 'jump_models.pkl'
if not pk_path.exists():
    raise SystemExit(f'jump_models.pkl not found at {pk_path}')

with open(pk_path, 'rb') as f:
    data = pickle.load(f)
print('Before keys:', list(data.keys()))

# Create simple pass-through scaler and classifier if missing
class SimpleScaler:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import numpy as _np
        return _np.asarray(X)

class SimpleClassifier:
    def predict(self, X):
        import numpy as _np
        X = _np.asarray(X)
        # default to no jumps
        return _np.zeros(X.shape[0], dtype=int)

if 'scaler' not in data:
    data['scaler'] = SimpleScaler()
if 'clf' not in data:
    data['clf'] = SimpleClassifier()

# Ensure boosters key exists
if 'boosters' not in data:
    data['boosters'] = {}

with open(pk_path, 'wb') as f:
    pickle.dump(data, f)

print('After keys:', list(data.keys()))

# Copy single LSTM files to normal/jump names if necessary
for tgt in ['likes', 'replies', 'views']:
    src = MODEL_DIR / f'uala_{tgt}_lstm.pt'
    dst1 = MODEL_DIR / f'uala_{tgt}_normal_lstm.pt'
    dst2 = MODEL_DIR / f'uala_{tgt}_jump_lstm.pt'
    if src.exists():
        if not dst1.exists():
            dst1.write_bytes(src.read_bytes())
            print('Wrote', dst1.name)
        else:
            print(dst1.name, 'already exists')
        if not dst2.exists():
            dst2.write_bytes(src.read_bytes())
            print('Wrote', dst2.name)
        else:
            print(dst2.name, 'already exists')
    else:
        print('Source LSTM missing:', src)

print('Done')
