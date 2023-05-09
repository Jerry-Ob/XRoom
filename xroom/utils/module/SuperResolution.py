from .FlowModule import FlowModule
import cv2
import numpy as np
import torch
from torch import nn


class RecursiveBlock(nn.Module):
    def __init__(self, num_channels: int, num_residual_unit: int) -> None:
        super(RecursiveBlock, self).__init__()
        self.num_residual_unit = num_residual_unit

        self.residual_unit = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        for _ in range(self.num_residual_unit):
            out = self.residual_unit(out)
            out = torch.add(out, x)

        return out


class DRRN(nn.Module):
    def __init__(self, num_residual_unit:int =9) -> None:
        super(DRRN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(1, 128, (3, 3), (1, 1), (1, 1), bias=False),
        )

        self.trunk = RecursiveBlock(128, num_residual_unit)
        self.conv2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 1, (3, 3), (1, 1), (1, 1), bias=False),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(identity, out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class SuperResolution(FlowModule):
    
    def __init__(self, weights='./', residual_layers = 9, gpu=False):
        self.model = DRRN(residual_layers)
        device = 'cuda' if gpu else 'cpu'
        self.model.load_state_dict(torch.load(weights,
                                              map_location=device)["state_dict"])
    
    
    def forward(self, images, *args, **kwargs):
        results = []
        for image in images:
            image = np.copy(image)
            image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            lr_y_image, lr_cr_image, lr_cb_image = cv2.split(image_ycbcr)
            image_torch = torch.stack([torch.from_numpy(np.transpose(lr_y_image.astype(np.float32)))])
            with torch.no_grad():
                out = self.model(image_torch).clamp_(0, 1.0)
            sr_y_image = out.permute([0, 2, 1]).numpy()[0]
            sr_y_image = sr_y_image.astype(np.float32) / 255.0
            image_ycbcr = cv2.merge([lr_y_image, lr_cr_image, lr_cb_image])
            image = cv2.cvtColor(image_ycbcr, cv2.COLOR_YCrCb2BGR)
            results.append(image)
        return [results]