import numpy as np
import h5py
import os

# -------------------------------
# Convert NPZ → H5
# -------------------------------

def npz_to_h5(npz_path, h5_path, compression=None):
    npz_data = np.load(npz_path)

    with h5py.File(h5_path, "w") as h5f:
        for key in npz_data.files:
            h5f.create_dataset(key, data=npz_data[key], compression=compression)

    return npz_data  


# -------------------------------
# Convert H5 → NPZ
# -------------------------------

def h5_to_npz(h5_path, npz_path):
    with h5py.File(h5_path, "r") as h5f:
        data_dict = {key: h5f[key][()] for key in h5f.keys()}  
    
    np.savez(npz_path, **data_dict)
    return data_dict


# -------------------------------
# Compare two NPZ files
# -------------------------------

def compare_npz(npz1, npz2):
    keys1 = set(npz1.files)
    keys2 = set(npz2.files)

    if keys1 != keys2:
        print("❌ Key mismatch:", keys1, keys2)
        return False

    for key in keys1:
        arr1 = npz1[key]
        arr2 = npz2[key]

        if arr1.shape != arr2.shape:
            print(f"❌ Shape mismatch in '{key}': {arr1.shape} vs {arr2.shape}")
            return False
        
        if arr1.dtype != arr2.dtype:
            print(f"❌ Dtype mismatch in '{key}': {arr1.dtype} vs {arr2.dtype}")
            return False
        
        if not np.array_equal(arr1, arr2):
            print(f"❌ Value mismatch in '{key}'")
            return False

    print("✅ All datasets match exactly!")
    return True


# -------------------------------
# Main process
# -------------------------------

if __name__ == "__main__":
    
    # original_npz_path = "E:/propnet-data/ori_data_arrays.npz"

    h5_path = "E:/propnet-data/data_arrays.h5"
    new_npz_path = "E:/propnet-data/data_arrays.npz"

    # print("🚀 Step 1: NPZ → H5")
    # orig_npz_data = npz_to_h5(original_npz_path, h5_path, compression="gzip")

    # print("🚀 Step 2: H5 → NPZ")
    new_npz_data = h5_to_npz(h5_path, new_npz_path)

    # print("🚀 Step 3: Verification...")
    # compare_npz(orig_npz_data, np.load(new_npz_path))


