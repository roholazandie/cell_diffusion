import numpy as np
from datasets import load_from_disk
import matplotlib.pyplot as plt
import os

dataset_path = "embryo"  # or the path where you saved it
dataset = load_from_disk(dataset_path)

save_path = "raw_embryo_images"

# Example: pick a sample
example_item = dataset[200]
gene_name = example_item["gene"]
xy_expr_matrix = example_item["matrix"]  # shape: [n_spots, 3]

print("Gene:", gene_name)
print("Matrix shape (n_spots x 3):", len(xy_expr_matrix), "x", len(xy_expr_matrix[0]))

spatial_coords = np.array([(x, y) for x, y, _ in xy_expr_matrix])
expression_values = np.array([expr for _, _, expr in xy_expr_matrix])

plt.figure(figsize=(6, 5))
plt.scatter(
    spatial_coords[:, 0],
    spatial_coords[:, 1],
    c=expression_values,
    cmap='viridis',
    s=10,
    edgecolor='none'
)
plt.axis('off')
plt.title(gene_name)
plt.gca().invert_yaxis()

filename = os.path.join(save_path, f"{gene_name}.png")
plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.close()
