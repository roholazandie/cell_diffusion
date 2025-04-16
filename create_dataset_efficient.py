import os
import re
import gc
import json
import scanpy as sc
import numpy as np
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, Value, Array3D

largest_image = {
    "min_x": 1,
    "max_x": 361,
    "min_y": -738,
    "max_y": -182,
    "width": 359,
    "height": 556
}
square_size = int(np.ceil(max(largest_image["width"], largest_image["height"]))) + 100
print(f"Using square_size = {square_size}") #656

save_directory = "data/images"
input_directory = "/media/rohola/ssd_storage/downloads"
filename_pattern = re.compile(r"E(?P<day>\d+\.\d+)_E(?P<Embryo>\d+)S(?P<Slice>\d+)")

# Sparse gene-level schema: only index + value of nonzero expressions
features = Features({
    "gene": Value("string"),
    "day": Value("float32"),
    "embryo": Value("int32"),
    "slice": Value("int32"),
    "matrix": Sequence(Sequence(Value("float32")))  # [spot_index, expression_value]
})

files = [f for f in os.listdir(input_directory) if f.endswith(".h5ad")]

for file in files:
    match = filename_pattern.search(file)
    if not match:
        print(f"Skipping file due to naming mismatch: {file}")
        continue

    day = float(match.group("day"))
    embryo = int(match.group("Embryo"))
    slice_ = int(match.group("Slice"))

    data_path = os.path.join(input_directory, file)
    print(f"Reading {file}")
    adata = sc.read_h5ad(data_path, backed="r")
    spatial_coords = adata.obsm["spatial"]

    min_x = spatial_coords[:, 0].min()
    max_x = spatial_coords[:, 0].max()
    min_y = spatial_coords[:, 1].min()
    max_y = spatial_coords[:, 1].max()

    current_center_x = (min_x + max_x) / 2
    current_center_y = (min_y + max_y) / 2
    global_center_x = (largest_image["min_x"] + largest_image["max_x"]) / 2
    global_center_y = (largest_image["min_y"] + largest_image["max_y"]) / 2

    shift_x = global_center_x - current_center_x + (square_size / 2 - global_center_x)
    shift_y = global_center_y - current_center_y + (square_size / 2 - global_center_y)

    # Apply coordinate shift and save global spatial index
    shifted_coords = np.column_stack([
        spatial_coords[:, 0] + shift_x,
        spatial_coords[:, 1] + shift_y
    ])

    output_path = os.path.join(save_directory, f"{file.replace('.h5ad', '')}")
    if os.path.exists(output_path):
        print(f"Directory already exists, skipping: {output_path}")
        continue
    os.makedirs(output_path, exist_ok=True)

    # Save shared spatial positions
    np.save(os.path.join(output_path, "spatial_coords.npy"), shifted_coords.astype(np.float32))
    print(f"Saved global spatial_coords for {file}: shape = {shifted_coords.shape}")

    # Save each gene's sparse expression
    for i, gene in enumerate(tqdm(adata.var_names, desc=f"Processing {file}")):
        try:
            expression_values = adata[:, i].X
            if hasattr(expression_values, "toarray"):
                expression_values = expression_values.toarray().flatten()
            else:
                expression_values = np.asarray(expression_values).flatten()

            # Keep only non-zero values
            nonzero_mask = expression_values != 0
            indices = np.nonzero(nonzero_mask)[0]
            values = expression_values[nonzero_mask]

            if len(indices) == 0:
                continue  # Skip genes with no expression

            sparse_matrix = [[int(idx), float(val)] for idx, val in zip(indices, values)]

            record = [{
                "gene": gene,
                "day": day,
                "embryo": embryo,
                "slice": slice_,
                "matrix": sparse_matrix
            }]

            single_dataset = Dataset.from_list(record, features=features)
            gene_path = os.path.join(output_path, f"{gene}.hf")
            single_dataset.save_to_disk(gene_path)

            del expression_values, sparse_matrix, single_dataset
            gc.collect()

        except Exception as e:
            print(f"Error processing gene {gene}: {e}")

    del adata
    gc.collect()
    print(f"Finished and saved: {file}")
