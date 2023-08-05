from algorithms import GapDenoise
from utils import Opt
import hdf5storage
import numpy as np
from loguru import logger
import cv2

def load_mask(file_path):
    mask = hdf5storage.loadmat(file_path)['mask']
    mask = np.transpose(mask, (1, 2, 0))
    logger.debug(mask.shape)
    return mask

def load_measure(measure_path):
    measure = cv2.imread(measure_path, -1)
    measure = measure[:, 400:2448]
    logger.debug(measure.shape)
    # cv2.imshow("measure", cv2.resize(measure, (1000, 1000)))
    # cv2.waitKey(0)
    return measure

def load_hsi(hsi_path):
    pass

def show_hsi(hsi):
    pass

def main():
    mask_path = "./mask/mask_0412_2048_2048_0_400_div_3.mat"
    image_path = "./image/0416_1.bmp"
    opt = Opt()
    opt.acc = True
    opt.iter = 500
    opt.nframe = 1
    opt.MAXB = 1
    opt.denoiser = "tv"
    opt.tviter = 3
    opt.tvweight = 0.01 * 1/opt.MAXB
    opt.rate = 1
    opt.flag_iqa = False
    algorithm = GapDenoise(opt)
    mask = load_mask(file_path)

    measure = load_measure(image_path)
    result = GapDenoise(mask, measure)

if __name__=='__main__':
    # load_mask("./mask/mask_0412_2048_2048_0_400_div_3.mat")
    load_measure("./image/0416_1.bmp")