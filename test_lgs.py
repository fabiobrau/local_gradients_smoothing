from configparser import ConfigParser
from torchvision.transforms import PILToTensor, ToPILImage
from PIL import Image
import torch
from lgs import LGS


def test():
    img = Image.open('test_image.jpg')
    cfg = ConfigParser()
    cfg.read('defaults.ini')
    loc_grad_smooth = LGS(**cfg['DEFAULT'])
    img_t = PILToTensor()(img) / 255.
    norm_grad = loc_grad_smooth.normalized_grad(img_t.unsqueeze(0))
    norm_grad.squeeze_()
    collage_t = torch.cat([img_t, norm_grad], dim=-1)
    collage = ToPILImage()(collage_t)
    collage.show()


if __name__ == '__main__':
    test()
