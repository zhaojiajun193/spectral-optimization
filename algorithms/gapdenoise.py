from utils import Opt
from utils import InferenceUtils
import numpy as np
from denoiser.denoiser import TV_denoising
from loguru import logger
class GapDenoise:
    def __init__(self, opt:Opt):
        self._opt = opt

    @property
    def opt(self):
        return self._opt

    def __call__(self, mask, measure):
        #调用gap-tv求解
        v = InferenceUtils.At_(np.expand_dims(measure, axis=2), mask)
        logger.debug(v.shape)
        # y1 = np.zeros(measure.shape)
        for i in range(self.opt.iter):
            yb = InferenceUtils.A_(v, mask)
            # logger.debug(yb.shape)
            if self.opt.acc:
                # y1 = y1 + (measure - yb)
                # logger.debug(y1.shape)
                x = v + self.opt.rate*(InferenceUtils.At_(np.expand_dims((measure-yb)/self.opt.Phisum, axis=-1) , mask))
            if self.opt.denoiser == "tv":
                # v = self.tv_denoising(v)
                noisyim = x
                noisyim = np.expand_dims(noisyim, axis=-1)
                v = TV_denoising(noisyim, lambda_val=self.opt.tvweight, iter=self.opt.tviter)

        return x

    # def tv_denoising(self, v):
    #     def dvt(x):
    #         y = np.concatenate(([-x[0]], -np.diff(x, axis=0), [x[-1]]))
    #         return y
    #     def dht(x):
    #         y = np.concatenate(([-x[:, 0]], -np.diff(x, axis=1), [x[:, -1]]))
    #         return y
    #     def clip(x, lambda_val):
    #         y = np.sign(x) * (np.minimum(np.abs(x), lambda_val))
    #         return y
    #     def dv(x):
    #         y = np.diff(x, axis=0)
    #         return y
    #     def dh(x):
    #         y = np.diff(x, axis=1)
    #         return y
    #     alpha = 5
    #     for it in range(self.opt.tviter):
    #         if it == 0:
    #             h, v = v.shape
    #             zh = np.zeros(h, v-1)
    #             zv = np.zeros(h-1, v)
    #         x0h = y0 - dht(zh)
    #         x0v = y0 - dvt(zv)
    #         x0 = (x0h + x0v)/2
    #         zh = clip(zh + 1/alpha*dh(x0), self.opt.tvweight/2)
    #         zv = clip(zv + 1/alpha*dv(x0), self.opt.tvweight/2)
    #     return x0