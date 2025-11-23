import h5py
import numpy as np
import os

def fix_h5_shape(path):
    """
    Checks an H5 file and ensures the 'features' dataset has shape (1, D).
    If it is (D,) or (N, D), it reshapes or mean-pools it.
    """
    try:
        with h5py.File(path, 'r+') as f:
            if 'features' not in f:
                return False, "No 'features' key found"
            
            arr = f['features'][:]
            
            # Case 1: Already correct (1, D)
            if arr.ndim == 2 and arr.shape[0] == 1:
                return True, "Already correct"
            
            # Case 2: Flat vector (D,) -> Reshape to (1, D)
            elif arr.ndim == 1:
                new_arr = arr.reshape(1, -1).astype(np.float32)
            
            # Case 3: Multiple patches (N, D) -> Mean pool to (1, D)
            elif arr.ndim == 2 and arr.shape[0] > 1:
                new_arr = arr.mean(axis=0, keepdims=True).astype(np.float32)
            
            else:
                return False, f"Unexpected shape {arr.shape}"

            # Atomic replacement
            del f['features']
            f.create_dataset('features', data=new_arr)
            return True, f"Fixed shape from {arr.shape} to {new_arr.shape}"
            
    except Exception as e:
        return False, f"Error: {e}"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)