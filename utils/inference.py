import numpy as np
import math


class InferenceUtils:
    @staticmethod
    def A_(x, Phi):
        '''
        forward model of snapshot compressive imaging(SCI)
        '''
        # for 3-D measurements
        return np.sum(x*Phi, axis=2)

    @staticmethod
    def At_(y, Phi):
        '''
        Transpose of the forward model
        '''
        return y*Phi