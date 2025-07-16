import numpy as np
from pathlib import Path

def compare_weights(run1_dir, run2_dir):
    w1 = np.load(Path(run1_dir) / "model_weights.npy", allow_pickle=True).item()
    w2 = np.load(Path(run2_dir) / "model_weights.npy", allow_pickle=True).item()

    mismatches = []
    for k in w1:
        if k not in w2:
            mismatches.append(f"{k} missing in run2")
            continue
        if not np.allclose(w1[k], w2[k], rtol=1e-5, atol=1e-8):
            mismatches.append(f"{k} differs")

    if mismatches:
        print("**** Mismatched weights found:")
        for m in mismatches:
            print("-", m)
    else:
        print("---- Weights match exactly!")

if __name__ == "__main__":
    compare_weights("repro_check_outputs/run1", "repro_check_outputs/run2")
