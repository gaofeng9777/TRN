import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image


# Custom dataset class
class HyperspectralOpticalFlowDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.spatial_stream_dir = os.path.join(data_dir, "spatial_stream")
        self.temporal_stream_dir = os.path.join(data_dir, "temporal_stream")

        # Get the list of file names for spatial and temporal streams
        self.spatial_files = sorted(os.listdir(self.spatial_stream_dir))
        self.temporal_files = sorted(os.listdir(self.temporal_stream_dir))

        # Ensure the number of spatial and temporal files match
        assert len(self.spatial_files) == len(self.temporal_files), \
            "Mismatch between spatial and temporal data files!"

    def __len__(self):
        return len(self.spatial_files)

    def __getitem__(self, idx):
        # Load spatial stream data (hyperspectral)
        spatial_path = os.path.join(self.spatial_stream_dir, self.spatial_files[idx])
        spatial_data = np.load(spatial_path)  # Assuming .npy format for hyperspectral data

        # Load temporal stream data (optical flow)
        temporal_path = os.path.join(self.temporal_stream_dir, self.temporal_files[idx])
        temporal_data = np.load(temporal_path)  # Assuming .npy format for optical flow data

        # Apply transforms if specified
        if self.transform:
            spatial_data = self.transform(spatial_data)
            temporal_data = self.transform(temporal_data)

        # Convert to tensors
        spatial_data = torch.tensor(spatial_data, dtype=torch.float32)
        temporal_data = torch.tensor(temporal_data, dtype=torch.float32)

        # Load label (assuming labels are stored as part of the file name: e.g., "data_0_label_1.npy")
        label = int(self.spatial_files[idx].split("_")[-1].split(".")[0])  # Extract label from file name

        return spatial_data, temporal_data, label


# Data preparation function
def prepare_data(data_dir, batch_size=16, shuffle=True, num_workers=4):
    """
    Prepares DataLoader for training and validation.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation datasets.
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 16, std=[0.5] * 16)  # Normalize for 16 channels
    ])

    # Create dataset
    dataset = HyperspectralOpticalFlowDataset(data_dir=data_dir, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    data_dir = "./data"  # Path to your data folder
    batch_size = 16

    train_loader, val_loader = prepare_data(data_dir, batch_size)

    # Example: Iterate through training data
    for spatial_data, temporal_data, label in train_loader:
        print(f"Spatial Data Shape: {spatial_data.shape}")  # (batch_size, 16, H, W)
        print(f"Temporal Data Shape: {temporal_data.shape}")  # (batch_size, 16, H, W)
        print(f"Labels: {label}")  # (batch_size,)
        break
