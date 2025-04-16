import os
from typing import Generic

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_from_disk
import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
import numpy as np

class GeneExpressionDataset(Dataset):
    def __init__(self, folder_path: str, square_size: int = 656):
        """
        Args:
            folder_path: Path to a folder
            square_size: Expected width/height of the image
        """
        self.folder_path = folder_path
        self.square_size = square_size

        # Load spatial coordinates once
        self.spatial_coords = np.load(os.path.join(folder_path, "spatial_coords.npy"))
        self.n_spots = self.spatial_coords.shape[0]

        # Load the combined HuggingFace dataset
        self.dataset = load_from_disk(os.path.join(folder_path, "combined_dataset.hf"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        gene_record = self.dataset[idx]

        expression_map = np.zeros(self.n_spots, dtype=np.float32)
        if gene_record["matrix"]:  # make sure it's not empty
            spot_indices, values = zip(*gene_record["matrix"])
            expression_map[np.array(spot_indices, dtype=int)] = values

        img = np.full((self.square_size, self.square_size), np.nan, dtype=np.float32)  # nan = unknown
        xs = np.clip(self.spatial_coords[:, 0].astype(int), 0, self.square_size - 1)
        ys = np.clip(self.spatial_coords[:, 1].astype(int), 0, self.square_size - 1)

        for x, y, e in zip(xs, ys, expression_map):
            img[y, x] = e  # measured spot (including 0s)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        return img_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = GeneExpressionDataset("data/E11.5_E1S4")

    # get the first gene image + its name
    output_dir = "outputs/tensors"
    for i, image_tensor in enumerate(dataset):
        print(image_tensor.shape)
        img = image_tensor.squeeze(0).numpy()

        plt.figure(figsize=(5, 5))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')  # show background separately
        plt.imshow(img, cmap=cmap, origin="upper", vmin=0, vmax=np.nanmax(img))
        # plt.title(f"Gene: {gene_record['gene']}")
        plt.colorbar(label="Expression")
        plt.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"outputs/tensors/{i}.png", dpi=300)

    # plt.imshow(image_tensor.squeeze(0).numpy(), cmap="viridis", vmin=0, vmax=1)
    # plt.title("Raw Expression Tensor")
    # plt.colorbar()
    # plt.show()