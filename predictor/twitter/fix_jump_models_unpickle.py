import pickle
from pathlib import Path
from predictor.twitter.model_placeholders import SimpleScaler, SimpleClassifier


class FixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map previously pickled classes that lived in __main__ to our importable placeholders
        if module == '__main__' and name == 'SimpleScaler':
            return SimpleScaler
        if module == '__main__' and name == 'SimpleClassifier':
            return SimpleClassifier
        return super().find_class(module, name)


def main():
    p = Path(__file__).resolve().parent / 'models_uala_v3_jumpboosted' / 'jump_models.pkl'
    print('Updating', p)
    with p.open('rb') as f:
        d = FixUnpickler(f).load()
    print('Old keys:', list(d.keys()))

    # Ensure safe importable placeholders
    d['scaler'] = SimpleScaler()
    d['clf'] = SimpleClassifier()
    d.setdefault('boosters', {})

    with p.open('wb') as f:
        pickle.dump(d, f)
    print('New keys:', list(d.keys()))


if __name__ == '__main__':
    main()
