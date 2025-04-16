import os
import re
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, Value

# use the find_largest_embryo function to get the largest object info
largest_embryo = {
    "min_x": 1,
    "max_x": 361,
    "min_y": -738,
    "max_y": -182,
    "width": 359,
    "height": 556
}

square_size = int(np.ceil(max(largest_embryo["width"], largest_embryo["height"])))
square_size += 100  # Add some padding
print(f"Using square_size = {square_size}") # Using square_size = 656

save_directory = "embryos"
input_directory = "/media/rohola/ssd_storage/downloads"
filename_pattern = re.compile(r"E(?P<day>\d+\.\d+)_E(?P<Embryo>\d+)S(?P<Slice>\d+)")
features = Features({
    "gene": Value("string"),
    "day": Value("float32"),
    "embryo": Value("int32"),
    "slice": Value("int32"),
    "matrix": Sequence(Sequence(Value("float32")))
})

# List all .h5ad files
files = [f for f in os.listdir(input_directory) if f.endswith(".h5ad")]

for file in files:
    # check if the

    match = filename_pattern.search(file)
    if not match:
        print(f"Skipping file due to naming mismatch: {file}")
        continue

    day = float(match.group("day"))
    embryo = int(match.group("Embryo"))
    slice_ = int(match.group("Slice"))

    data_path = os.path.join(input_directory, file)
    adata = sc.read_h5ad(data_path)
    spatial_coords = adata.obsm["spatial"]

    # Current file's bounding box
    min_x = spatial_coords[:, 0].min()
    max_x = spatial_coords[:, 0].max()
    min_y = spatial_coords[:, 1].min()
    max_y = spatial_coords[:, 1].max()
    width = max_x - min_x
    height = max_y - min_y

    # Compute center of current and largest object
    current_center_x = (min_x + max_x) / 2
    current_center_y = (min_y + max_y) / 2
    global_center_x = (largest_embryo["min_x"] + largest_embryo["max_x"]) / 2
    global_center_y = (largest_embryo["min_y"] + largest_embryo["max_y"]) / 2

    # Compute shift to center current object in the global square
    shift_x = global_center_x - current_center_x + (square_size / 2 - global_center_x)
    shift_y = global_center_y - current_center_y + (square_size / 2 - global_center_y)

    # Prepare all gene records
    records = []
    is_sparse = hasattr(adata.X, "toarray")

    for i, gene in enumerate(tqdm(adata.var_names, desc=f"Processing {file}")):
        expression_values = adata.X[:, i].toarray().flatten() if is_sparse else adata.X[:, i].flatten()

        new_coords = []
        for (x, y), expr in zip(spatial_coords, expression_values):
            x_new = x + shift_x
            y_new = y + shift_y

            # Warning if something is out of bounds (shouldn't happen!)
            if not (0 <= x_new < square_size and 0 <= y_new < square_size):
                print(f"Warning: ({x_new:.1f}, {y_new:.1f}) out of bounds for {file}, gene {gene}")

            new_coords.append([x_new, y_new, expr])

        # if i % 50 == 0:
        #     xs, ys, es = zip(*new_coords)
        #     plt.figure(figsize=(6, 6))
        #     plt.scatter(xs, ys, c=es, cmap="viridis", s=20)
        #     plt.title(f"{gene} (Shifted to Center)")
        #     plt.xlim(0, square_size)
        #     plt.ylim(0, square_size)
        #     plt.gca().invert_yaxis()
        #     plt.colorbar(label="expression")
        #     plt.tight_layout()
        #     plt.show()
        #     break

        records.append({
            "gene": gene,
            "day": day,
            "embryo": embryo,
            "slice": slice_,
            "matrix": new_coords
        })

    dataset = Dataset.from_list(records, features=features)
    output_path = os.path.join(save_directory, f"{file.replace('.h5ad', '')}")
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    print(f"Saved: {output_path}")
