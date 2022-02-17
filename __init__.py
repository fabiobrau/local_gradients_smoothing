from configparser import ConfigParser
from .lgs import LGS

cfg = ConfigParser()
cfg.read('defaults.ini')
LGS_GradientMask = LGS(cfg['DEFAULT'])
