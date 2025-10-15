import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM(nn.Module):
    """Max-Feature-Map activation"""
    def __init__(self, in_channels):
        super(MFM, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        return torch.max(x[:, 0:self.in_channels//2, ...], 
                        x[:, self.in_channels//2:, ...])


class LightCNNOriginal(nn.Module):
    """LightCNN architecture for voice anti-spoofing"""
    def __init__(self, input_shape=(1, 863, 600), num_classes=2, dropout_prob=0.75):
        super(LightCNNOriginal, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MFM(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv2a = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.mfm2a = MFM(64)
        self.bn2a = nn.BatchNorm2d(32)

        self.conv2b = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        self.mfm2b = MFM(96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2b = nn.BatchNorm2d(48)

        # Conv Block 3
        self.conv3a = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0)
        self.mfm3a = MFM(96)
        self.bn3a = nn.BatchNorm2d(48)

        self.conv3b = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.mfm3b = MFM(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 4
        self.conv4a = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.mfm4a = MFM(128)
        self.bn4a = nn.BatchNorm2d(64)

        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mfm4b = MFM(64)
        self.bn4b = nn.BatchNorm2d(32)

        self.conv4c = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.mfm4c = MFM(64)
        self.bn4c = nn.BatchNorm2d(32)

        self.conv4d = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm4d = MFM(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate feature dimension
        self.feature_dim = self._get_feature_dim(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_dim, 160)
        self.mfm_fc1 = MFM(160)
        self.bn_fc1 = nn.BatchNorm1d(80)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.fc2 = nn.Linear(80, num_classes)

    def _get_feature_dim(self, input_shape):
        """Calculate the feature dimension after conv layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.features(dummy_input)
            return x.view(1, -1).size(1)

    def features(self, x):
        """Extract features through conv layers"""
        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.bn2a(self.mfm2a(self.conv2a(x)))
        x = self.bn2b(self.pool2(self.mfm2b(self.conv2b(x))))
        x = self.bn3a(self.mfm3a(self.conv3a(x)))
        x = self.pool3(self.mfm3b(self.conv3b(x)))
        x = self.bn4a(self.mfm4a(self.conv4a(x)))
        x = self.bn4b(self.mfm4b(self.conv4b(x)))
        x = self.bn4c(self.mfm4c(self.conv4c(x)))
        x = self.pool4(self.mfm4d(self.conv4d(x)))
        return x

    def forward(self, x):
        """Forward pass"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.bn_fc1(self.mfm_fc1(self.fc1(x))))
        x = self.fc2(x)
        return x
