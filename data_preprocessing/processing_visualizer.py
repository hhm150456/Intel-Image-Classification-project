import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


########################################################################
### Note that visualize_before_after was made using ChatGPT ############
########################################################################

class Visualizer:
    def __init__(self, dataset):
        self.dataset = dataset

    def visualize_before_after(self, index=0):
        original_img_path, label = self.dataset.samples[index]  # Get image path
        original_img = Image.open(original_img_path).convert("RGB")  # Open as RGB

        # Get preprocessed image
        preprocessed_img, _ = self.dataset[index]  # Already transformed by dataset

        # Convert PIL image to NumPy array for histogram
        original_img_np = np.array(original_img)

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))  # 2x2 grid

        # Display Original Image
        ax[0, 0].imshow(original_img)
        ax[0, 0].set_title(f"Original Image (Label: {label})")
        ax[0, 0].axis("off")

        # Display Preprocessed Image
        ax[0, 1].imshow(preprocessed_img.permute(1, 2, 0))  # Convert tensor to image
        ax[0, 1].set_title("Preprocessed Image")
        ax[0, 1].axis("off")

        # Histogram of Original Image
        ax[1, 0].hist(original_img_np.ravel(), bins=50, density=True, color='blue', alpha=0.7)
        ax[1, 0].set_title("Original Image Histogram")
        ax[1, 0].set_xlabel("Pixel Value")
        ax[1, 0].set_ylabel("Frequency")

        # Histogram of Preprocessed Image
        ax[1, 1].hist(preprocessed_img.numpy().ravel(), bins=50, density=True, color='red', alpha=0.7)
        ax[1, 1].set_title("Preprocessed Image Histogram")
        ax[1, 1].set_xlabel("Pixel Value")
        ax[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()