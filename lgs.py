import torch
from typing import Union
from torchvision.transforms import ToTensor
from PIL.Image import Image
from PIL.ImageOps import grayscale
import gradient


class LGS:
    def __init__(self, window_size: int,
                 overlap: int,
                 smoothing_factor: float,
                 threshold: float,
                 grad_method: str):
        self.window_size = int(window_size)
        self.overlap = int(overlap)
        self.smoothing_factor = float(smoothing_factor)
        self.threshold = float(threshold)
        self.to_tensor = ToTensor()

        self.grad = getattr(gradient, grad_method)()
        self.stride = self.window_size - self.overlap
        self.fold = torch.nn.functional.fold
        self.unfold = torch.nn.functional.unfold

    def normalized_grad(self, img: torch.Tensor) -> torch.Tensor:
        img_grad = self.grad(img)
        max_grad = torch.amax(img_grad, dim=(2, 3), keepdim=True)
        min_grad = torch.amin(img_grad, dim=(2, 3), keepdim=True)
        img_grad = (img_grad - min_grad) / (max_grad - min_grad + 1e-7)
        return img_grad

    def get_mask(self, img_t: torch.tensor) -> torch.Tensor:
        grad = self.normalized_grad(img_t)
        grad_unfolded = self.unfold(
            grad, self.window_size, stride=self.stride)
        mask_unfolded = torch.mean(
            grad_unfolded, dim=1, keepdim=True) > self.threshold
        mask_unfolded = mask_unfolded.repeat(1, grad_unfolded.shape[1], 1)
        mask_unfolded = mask_unfolded.float()
        mask_folded = self.fold(
            mask_unfolded, grad.shape[2:], kernel_size=self.window_size, stride=self.stride)
        mask_folded = (mask_folded >= 1).float()
        grad *= mask_folded
        grad = torch.clamp(self.smoothing_factor * grad, 0, 1)
        return grad

    def __call__(self, img: Union[torch.Tensor, Image]):
        if isinstance(img, Image):
            img_t = self.pil_to_tensor(img)
        elif isinstance(img, torch.Tensor):
            img_t = img.clone()
        else:
            raise NotImplementedError(
                f'Type of {type(img)} not supported yet.')
        mask = self.get_mask(img_t)
        return mask

    @staticmethod
    def pil_to_tensor(img: Image) -> torch.Tensor:
        img_gray = grayscale(img)
        img_t = ToTensor()(img_gray)
        img_t.unsqueeze_(0)
        return img_t
