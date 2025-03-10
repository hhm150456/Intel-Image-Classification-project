import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from PIL import Image


class DataPreprocessor:
    """
    The class is responsiple of all preprocessing Stages: DataAugmentation data loader and visualization

    """
    def __init__(self, data_path, test_data_path, transforms=None, batch_size=32, train_ratio=0.8, val_ratio=0.2):
        """
        The data will be splitted into: 70% training, 15% Validation, 15% testing
        Constructor Variables:
            - data_path: getting the dataset path directory 
            - batch_size: how much of data will be dealed in one cycle 
            - train_ratio, val_ration and testing one are 70 15 15 respectivly

        """
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        """
        TRansformation and Data Augmentation part:
            - change the size of data to be all in 128*128
            - rotating the images by 45 degree
            - flipping images horizontally during training
            - transfer the images into tensors
            - normalizing the data with mean, std (values were chosen by taking popular choices)
        references:
             - https://pytorch.org/vision/master/auto_examples/transforms/plot_transforms_getting_started.html
             - https://pytorch.org/vision/stable/transforms.html

        """
        if transforms is None:
            transforms = v2.Compose([
                v2.Resize((128, 128)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=45),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.dataset = datasets.ImageFolder(root=self.data_path, transform= transforms)
        self.test_dataset = datasets.ImageFolder(root=self.test_data_path, transform=transforms)

    
    def prepare_data(self):

        """
        start to split the data into training, validation, testing
        train_size = 0.7 * length of the whole data 
        and do on for the other variables

        Data loader stage to split the data and load it in effecient way
        shuffle: True for training 
        shuffle: False for validation and testing(no need for training)

        reference:
            - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        """
        total_size = len(self.dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset  = random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = self.batch_size, shuffle = False)
        test_loader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)
        
        return train_loader, val_loader, test_loader
    
    
########################################################################
### Note that visualize_before_after was amde using ChatGPT ############
########################################################################
    
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



