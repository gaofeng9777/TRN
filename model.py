import torch
import torch.nn as nn

class TwoStreamConvNet(nn.Module):
    def __init__(self, num_classes=101):
        super(TwoStreamConvNet, self).__init__()

        # Spatial Stream (Single frame RGB)
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Temporal Stream (Optical flow)
        self.temporal_stream = nn.Sequential(
            nn.Conv2d(20, 96, kernel_size=7, stride=2, padding=3),  # 20 channels for stacked optical flow
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully Connected Layers (Shared)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, rgb_frame, optical_flow):
        # Process Spatial Stream
        spatial_out = self.spatial_stream(rgb_frame)
        spatial_out = spatial_out.view(spatial_out.size(0), -1)  # Flatten for FC layers

        # Process Temporal Stream
        temporal_out = self.temporal_stream(optical_flow)
        temporal_out = temporal_out.view(temporal_out.size(0), -1)  # Flatten for FC layers

        # Concatenate and pass through FC layers
        combined = torch.cat((spatial_out, temporal_out), dim=1)
        output = self.fc(combined)

        return output


if __name__ == "__main__":
    # Example usage
    model = TwoStreamConvNet(num_classes=101)

    # Example input data
    rgb_frame = torch.randn(8, 3, 224, 224)  # Batch of 8 RGB frames
    optical_flow = torch.randn(8, 20, 224, 224)  # Batch of 8 stacked optical flow

    # Forward pass
    output = model(rgb_frame, optical_flow)
    print("Output shape:", output.shape)  # Should be [8, 101]
