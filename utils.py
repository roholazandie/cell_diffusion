import os
import scanpy as sc
import json
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm


def find_largest_embryo(input_directory):

    files = [f for f in os.listdir(input_directory) if f.endswith(".h5ad")]

    all_coords = {}
    largest_area = 0
    largest_embryo = {}

    # Iterate through all files
    for file in tqdm(files):
        data_path = os.path.join(input_directory, file)
        adata = sc.read_h5ad(data_path)

        spatial_coordinates = adata.obsm["spatial"]

        min_x = spatial_coordinates[:, 0].min()
        max_x = spatial_coordinates[:, 0].max()
        min_y = spatial_coordinates[:, 1].min()
        max_y = spatial_coordinates[:, 1].max()

        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        all_coords[file] = [(int(min_x), int(max_x), int(min_y), int(max_y))]

        if area > largest_area:
            largest_area = area
            largest_embryo = {
                "file": file,
                "min_x": int(min_x),
                "max_x": int(max_x),
                "min_y": int(min_y),
                "max_y": int(max_y),
                "width": int(width),
                "height": int(height),
                "area": int(area),
            }

    # Output results
    print("\nLargest object info:")
    for k, v in largest_embryo.items():
        print(f"{k}: {v}")

    # Save per-file coordinate info
    json.dump(all_coords, open(os.path.join(input_directory, "all_coords.json"), "w"))

    return largest_embryo


def combine_hf_datasets(parent_dir):
    """
    Combine all gene-wise datasets into a single dataset for each embryo.
    :return:
    """
    # Path to the parent folder where each subdirectory contains gene-wise datasets


    # List all subdirectories
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    output_path = os.path.join(parent_dir, "combined")
    for subdir in tqdm(subdirs, desc="Aggregating embryo datasets"):
        dir_path = os.path.join(parent_dir, subdir)

        # Collect all paths to .hf datasets inside the subdir
        gene_datasets = []
        for fname in tqdm(os.listdir(dir_path), desc=f"Aggregating {subdir}"):
            gene_path = os.path.join(dir_path, fname)
            if os.path.isdir(gene_path) and "dataset_info.json" in os.listdir(gene_path):
                try:
                    dataset = load_from_disk(gene_path)
                    gene_datasets.append(dataset)
                except Exception as e:
                    print(f"Failed to load {gene_path}: {e}")

        if not gene_datasets:
            print(f"❌ No valid datasets in {dir_path}")
            continue

        try:
            # Combine them into a single dataset
            combined = concatenate_datasets(gene_datasets)
            combined.save_to_disk(os.path.join(output_path, f"{fname}.hf"))
            print(f"Combined dataset saved in {output_path}/{fname}.hf")
        except Exception as e:
            print(f"Failed to concatenate in {dir_path}: {e}")


if __name__ == "__main__":
    # find_largest_embryo(input_directory)
    parent_dir = "embryos"
    combine_hf_datasets(parent_dir)