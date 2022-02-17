from lgs.local_gradients_smoothing import LocalGradientsSmoothing
from lgs.gradient import Gradient, GradientSmooth
from configs import Configuration
cfg = Configuration()
get_lgs_mask = LocalGradientsSmoothing(**cfg.get('DEFAULT'))
