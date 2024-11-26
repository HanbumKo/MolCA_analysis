import gzip
import glob
import os
import numpy as np
import cupy as cp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from scipy.sparse import csr_matrix, vstack
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

def read_molecules_from_sdf_gz(file_path):
    """Generator function to read molecules from a compressed SDF file."""
    with gzip.open(file_path, 'rb') as f:
        suppl = Chem.ForwardSDMolSupplier(f)
        for mol in suppl:
            if mol is not None:
                yield mol

def compute_fingerprints(mol_iter):
    """Compute Morgan fingerprints for a set of molecules."""
    # fps = []
    fps = [np.zeros((2048,), dtype=np.bool_) for _ in range(500000)]
    for i, mol in enumerate(mol_iter):
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        cid = mol.GetProp("_Name") if mol.HasProp("_Name") else "N/A"
        print(cid)
        arr = np.zeros((0,), dtype=np.bool_)
        DataStructs.ConvertToNumpyArray(fp, arr)
        # fps.append(arr)
        if cid == "N/A":continue
        fps[int(cid)%500000-1]= arr
        if i == 100:break
    return csr_matrix(fps)

def compute_tanimoto_gpu(fps_i, fps_j, threshold=0.7):
    """
    Compute Tanimoto similarities between two sets of fingerprints using GPU.
    Only similarities above the threshold are returned.
    """
    # Convert to CuPy sparse matrices
    fps_i_gpu = cp.sparse.csr_matrix(fps_i, dtype=cp.float32)
    fps_j_gpu = cp.sparse.csr_matrix(fps_j, dtype=cp.float32)

    # Batch sizes (adjust based on GPU memory)
    batch_size_i = 1000
    batch_size_j = 100000

    num_i = fps_i_gpu.shape[0]
    num_j = fps_j_gpu.shape[0]

    results = []

    # Precompute the norms (number of bits set to 1)
    sum_i = cp.diff(fps_i_gpu.indptr).astype(cp.float32)
    sum_j = cp.diff(fps_j_gpu.indptr).astype(cp.float32)

    for i_start in tqdm(range(0, num_i, batch_size_i), desc="Computing similarities"):
        i_end = min(i_start + batch_size_i, num_i)
        fps_i_batch = fps_i_gpu[i_start:i_end]

        sum_i_batch = sum_i[i_start:i_end]

        for j_start in range(0, num_j, batch_size_j):
            j_end = min(j_start + batch_size_j, num_j)
            fps_j_batch = fps_j_gpu[j_start:j_end]

            sum_j_batch = sum_j[j_start:j_end]

            # Compute intersections
            intersections = fps_i_batch.dot(fps_j_batch.T).astype(cp.float32)

            # Compute unions
            unions = sum_i_batch[:, None] + sum_j_batch[None, :] - intersections

            # Compute Tanimoto similarity
            tanimoto_sim = intersections / unions

            # Apply threshold and store results
            mask = tanimoto_sim >= threshold
            tanimoto_sim = tanimoto_sim.multiply(mask)

            # Store the non-zero results
            coo = tanimoto_sim.tocoo()
            rows = coo.row + i_start
            cols = coo.col + j_start
            data = coo.data

            # Append to results
            results.append((rows.get(), cols.get(), data.get()))

    # Combine results
    if results:
        rows = np.concatenate([r[0] for r in results])
        cols = np.concatenate([r[1] for r in results])
        data = np.concatenate([r[2] for r in results])

        sim_matrix = csr_matrix((data, (rows, cols)), shape=(num_i, num_j))
    else:
        sim_matrix = csr_matrix((num_i, num_j))

    return sim_matrix

def process_file_pair(file_i, file_j, threshold):
    """Process a pair of files and compute similarities."""
    print(f"Processing files: {file_i} and {file_j}")

    # Read and compute fingerprints for file_i
    mol_iter_i = read_molecules_from_sdf_gz(file_i)
    fps_i = compute_fingerprints(mol_iter_i)

    # Read and compute fingerprints for file_j
    mol_iter_j = read_molecules_from_sdf_gz(file_j)
    fps_j = compute_fingerprints(mol_iter_j)

    # Compute similarities
    sim_matrix = compute_tanimoto_gpu(fps_i, fps_j, threshold)

    # Store results
    save_similarity_matrix(sim_matrix, file_i, file_j)

def save_similarity_matrix(sim_matrix, file_i, file_j):
    """Save the similarity matrix to a file."""
    output_dir = "similarities"
    os.makedirs(output_dir, exist_ok=True)
    base_i = os.path.basename(file_i)
    base_j = os.path.basename(file_j)
    output_file = os.path.join(output_dir, f"{base_i}_{base_j}_sims.npy")
    cp.save(output_file, sim_matrix)

def main_multiprocess():
    # Set the similarity threshold
    similarity_threshold = 0.7

    # Get the list of SDF files
    sdf_files = sorted(glob.glob('data/pubchem_dump/Compound_*.sdf.gz'))[:3]

    # Use multiprocessing to parallelize across CPU cores
    # num_workers = max(1, cpu_count() - 1)
    num_workers = 1
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx_i, file_i in enumerate(sdf_files):
            for file_j in sdf_files[idx_i:]:
                futures.append(executor.submit(process_file_pair, file_i, file_j, similarity_threshold))
        # Wait for all futures to complete
        for future in futures:
            future.result()


def main_singleprocess():
    # Set the similarity threshold
    similarity_threshold = 0.7

    # Get the list of SDF files
    sdf_files = sorted(glob.glob('data/pubchem_dump/Compound_*.sdf.gz'))[:3]

    # Process all file pairs
    for idx_i, file_i in enumerate(sdf_files):
        for file_j in sdf_files[idx_i:]:
            process_file_pair(file_i, file_j, similarity_threshold)

if __name__ == "__main__":
    main_singleprocess()
