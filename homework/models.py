import torch
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
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN.forward')


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
