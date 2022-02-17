import torch
from gradient import Gradient


class LGS:
    def __init__(self, window_size: int,
                 smoothing_factor: float, threshold: float):
        self.window_size = int(window_size)
        self.smoothing_factor = float(smoothing_factor)
        self.threshold = float(threshold)
        self.grad = Gradient()

    def normalized_grad(self, img: torch.Tensor) -> torch.Tensor:
        print(img.shape)
        img_grad = self.grad(img)
        max_pool = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        max_grad = max_pool(img_grad)
        min_grad = -max_pool(-img_grad)
        img_grad -= min_grad
        img_grad /= (max_grad - min_grad + 1e-7)
        return img_grad
