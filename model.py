import torch
import torch.nn as nn

class TwoStreamConvNet(nn.Module):
    def __init__(self, num_classes=101):
        super(TwoStreamConvNet, self).__init__()

        # Spatial Stream (Single frame RGB)
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(16, 96, kernel_size=7, stride=2, padding=3),
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
            nn.Conv2d(16, 96, kernel_size=7, stride=2, padding=3),  # 16 channels for stacked optical flow
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
            nn.Linear(512 * 16 * 16 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_frame, optical_flow):
        # Process Spatial Stream
        spatial_out = self.spatial_stream(rgb_frame)
        spatial_out = spatial_out.view(spatial_out.size(0), -1)  # Flatten for FC layers
        #print("Output shape:", spatial_out.shape)

        # Process Temporal Stream
        temporal_out = self.temporal_stream(optical_flow)
        temporal_out = temporal_out.view(temporal_out.size(0), -1)  # Flatten for FC layers
        #print("Output shape:", temporal_out.shape)

        # Concatenate and pass through FC layers
        combined = torch.cat((spatial_out, temporal_out), dim=1)
        output = self.fc(combined)

        return output

class ConvNet(nn.Module):
    def __init__(self, num_classes=101):
        super(ConvNet, self).__init__()

        # Spatial Stream (Hyperspectral frame 16)
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(16, 96, kernel_size=7, stride=2, padding=3),
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
            nn.Linear(512 * 16 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_frame):
        # Process Spatial Stream
        spatial_out = self.spatial_stream(rgb_frame)
        #print("Output shape:", spatial_out.shape)
        spatial_out = spatial_out.view(spatial_out.size(0), -1)  # Flatten for FC layers
        #print("Output shape:", spatial_out.shape)
        output = self.fc(spatial_out)

        return output


if __name__ == "__main__":
    #Example usage
    model = TwoStreamConvNet(num_classes=101)

    # Example input data
    rgb_frame = torch.randn(8, 16, 512, 512)  # Batch of 16 hyperspectral frames
    optical_flow = torch.randn(8, 16, 512, 512)  # Batch of 8 stacked optical flow

    # Forward pass
    output = model(rgb_frame, optical_flow)
    print("Output shape:", output.shape)  # Should be [8, 101]

    # model = ConvNet(num_classes=8)
    # rgb_frame = torch.randn(8, 16, 512, 512)  # Batch of 8 HyperSpectral frames
    # output = model(rgb_frame)
    # print("Output shape:", output.shape)  # Should be [8, 101]