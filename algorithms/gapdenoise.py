from utils import Opt
from utils import InferenceUtils

class GapDenoise:
    def __init__(self, opt:Opt):
        self._opt = opt

    @property
    def opt(self):
        return self._opt

    def __call__(self, mask, measure):
        #调用gap-tv求解
        pass