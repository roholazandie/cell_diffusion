from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset = load_from_disk("data/E11.5_E1S4/combined_dataset.hf")
spatial_coords = np.load("data/E11.5_E1S4/spatial_coords.npy")

# Configuration
square_size = 656

for gene_record in dataset:
    gene = gene_record["gene"]
    # Create dense array of zeros
    n_spots = spatial_coords.shape[0]
    expression_map = np.zeros(n_spots, dtype=np.float32)

    # Fill non-zero values
    spot_indices, values = zip(*gene_record["matrix"])
    expression_map[np.array(spot_indices, dtype=int)] = values

    # Plot
    xs, ys = spatial_coords[:, 0], spatial_coords[:, 1]
    es = expression_map

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c=es, cmap="viridis", s=1)
    plt.title(f"{gene} (Shifted to Center)")
    plt.xlim(0, square_size)
    plt.ylim(0, square_size)
    plt.gca().invert_yaxis()
    plt.colorbar(label="expression")
    plt.tight_layout()
    # plt.show()

    plt.savefig(f"outputs/{gene}.png")
