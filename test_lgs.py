from configparser import ConfigParser
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch
from lgs import LGS


def test():
    cfg = ConfigParser()
    cfg.read('defaults.ini')
    img_path = cfg['TESTING']
    img = Image.open(img_path)
    loc_grad_smooth = LGS(**cfg['DEFAULT'])
    grad_mask = loc_grad_smooth(img)
    grad_mask = grad_mask.repeat((3, 1, 1))
    img_t = ToTensor()(img)
    collage_t = torch.cat([img_t, grad_mask, img_t * (1 - grad_mask)], dim=-1)
    collage = ToPILImage()(collage_t)
    collage.show()


if __name__ == '__main__':
    test()
