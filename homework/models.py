import torch
import numpy as np
import torch.nn.functional as F


class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, stride):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, stride=stride),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class TBlock(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=2, stride=2),
            torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU())

    def forward(self, x):
        return self.net(x)


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[32,64,128], n_input_channels=3, n_classes=6):
        super().__init__()

        # First layers
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
             torch.nn.BatchNorm2d(32),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # Add Layers
        c = 32
        for l in layers:
            L.append(Block(c, l, stride=2))
            c = l

        # Put layers together
        self.network = torch.nn.Sequential(*L)

        # Append linear layer to the end
        self.classifier = torch.nn.Linear(layers[-1], n_classes)

    def forward(self, x):

        # Compute feature maps
        x = self.network(x)

        # Global average pooling (reduces height and width to a single number while retaining channels)
        x = x.mean(dim=[2, 3])

        return self.classifier(x)


class FCN(torch.nn.Module):
    def __init__(self, layers=[32,64,128], n_input_channels=3, n_classes=5, normalize=False):
        super().__init__()

        # Contracting path (Encoder)
        self.down0 = Block(n_input_channels, 5, stride=1)  # for identity
        self.down1 = Block(n_input_channels, layers[0], stride=1)
        self.down2 = Block(layers[0], layers[1], stride=1)
        self.down3 = Block(layers[1], layers[2], stride=1)
        self.down4 = Block(layers[0], 5, stride=1)  # for use with small images

        # Expansive path (Decoder)
        self.up1 = TBlock(layers[2], layers[1])
        self.conv1 = torch.nn.Conv2d(in_channels=layers[2], out_channels=layers[1], kernel_size=3, stride=1,
                                     padding=1)
        self.up2 = TBlock(layers[1], layers[0])
        self.conv2 = torch.nn.Conv2d(in_channels=layers[1], out_channels=layers[0], kernel_size=3, stride=1,
                                     padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3, stride=1,
                                     padding=1)
        self.up3 = TBlock(layers[0], n_classes)

        # Other layers
        # self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.max_pool = torch.nn.MaxPool2d(2)

    def forward(self, x):

        # For small images
        if x.size()[-1] <= 16 or x.size()[-2] <= 16:
            # Downsample and upsample once
            d1 = self.down1(x)
            d2 = self.down4(d1)

            return d2

        # Identity
        identity = x
        identity = self.down0(identity)
        # print(np.shape(identity))

        # Encoder (downsampling)
        d1 = self.down1(x)
        d2 = self.max_pool(d1)

        d3 = self.down2(d2)
        d4 = self.max_pool(d3)

        d5 = self.down3(d4)
        d6 = self.max_pool(d5)

        # Decoder (upsampling)
        e1 = self.up1(d6)
        s1 = torch.cat([e1, d4], dim=1)  # skip connection
        c1 = self.conv1(s1)

        e2 = self.up2(c1)
        s2 = torch.cat([e2, d2], dim=1)  # skip connection
        c2 = self.conv2(s2)

        # Add identity connection
        e3 = self.up3(c2)
        # print(np.shape(e3))
        s3 = torch.cat([e3, identity], dim=1)  # identity connection
        # print(np.shape(s3))
        c3 = self.conv3(s3)

        return c3


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
