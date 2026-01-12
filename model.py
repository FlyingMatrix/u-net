import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): 
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (N, C, H, W) -> (N, C, H/2, W/2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, 
                                               out_channels=feature, 
                                               kernel_size=2, 
                                               stride=2)) # (N, C, H, W) -> (N, C/2, H*2, W*2)
            self.ups.append(DoubleConv(in_channels=feature*2, out_channels=feature))

    def forward(self, x):
        skip_connections = []

        # Down part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottle neck
        x = self.bottleneck(x)

        # Reverse skip_connection after bottle neck
        skip_connections = skip_connections[::-1]

        # Up part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.resize(img=x, size=skip_connection.shape[2:]) # (H, W)
            
            concat = torch.cat(tensors=(skip_connection, x), dim=1) # (C)
            x = self.ups[idx+1](concat)

        # Final conv
        output = self.final_conv(x)

        return output

def test():
    input = torch.randn(size=(6, 1, 572, 572)) # (N, C, H, W)
    unet = UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
    output = unet(input)
    print(">>> Input shape: " + str(input.shape))
    print(">>> Output shape: " + str(output.shape))
    assert input.shape == output.shape

if __name__ == "__main__":
    test()
