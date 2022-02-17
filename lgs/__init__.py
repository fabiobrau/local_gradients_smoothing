from lgs.local_gradients_smoothing import LocalGradientsSmoothing
from lgs.gradient import Gradient, GradientSmooth
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('defaults.ini')
get_lgs_mask = LocalGradientsSmoothing(**cfg['DEFAULT'])
