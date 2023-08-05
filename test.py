from utils import ImageUtils, Opt
import cv2
from loguru import logger
from tests import SRnetDenoiseTest
import spectral
import os
import numpy as np
from algorithms import GapDenoise
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    image = cv2.imread("./image/0416_1.bmp", -1)
    image = image[:, 400:2448]
    image = ImageUtils.resize_image(image, 1000, 1000)
    cv2.imshow("test", image)
    cv2.waitKey(0)
    image = ImageUtils.load_mono_image("./image/0416_1.bmp")
    logger.debug(image.shape)
    srnetDenoiseTest = SRnetDenoiseTest("srnet", "./model_zoo/SRNet_lib.pth")
    hsi = srnetDenoiseTest.test_inference(image)
    logger.debug(hsi.shape)
    hsi_2 = ImageUtils.transpose_hsi(hsi)
    # spectral.imshow(hsi_2, (24, 15, 4))
    logger.debug(hsi_2.shape)
    ImageUtils.show_hsi(hsi_2, (24, 15, 4))
    rgb = ImageUtils.HSI2RGB_function(np.arange(400, 710, 10), hsi)
    rgb = (rgb*255).astype(np.uint8)
    cv2.imshow("test", ImageUtils.cvt_color_rgb_2_bgr(ImageUtils.resize_image(rgb, 1000, 1000)))
    cv2.waitKey(0)
    ImageUtils.save_rgb_image("./1.png", ImageUtils.cvt_color_bgr_2_rgb(rgb))

def test_gaptv_denoise():
    opt = Opt()
    opt.rate = 1
    opt.acc = True
    opt.flag_iqa = False
    opt.nframe = 1
    opt.MAXB = 1
    opt.denoiser = "tv"
    opt.iter = 500
    opt.tviter = 3
    opt.tvweight = 0.01
    #(2048, 2048, 61)
    mask = ImageUtils.load_mask_h5py("./mask/mask_0412_2048_2048_0_400.mat")
    #(2048, 2048)
    measure = ImageUtils.load_mono_image("./image/0416_1.bmp")
    measure = measure[-2048:, -2048:]
    start_location_x = 800
    start_location_y = 500
    stride = 2
    patch_size = 16
    image = ImageUtils.circle_point(measure, start_location_y, start_location_x)
    ImageUtils.save_rgb_image("./circle_image.png", image)
    measure = measure[start_location_x-patch_size : start_location_x + patch_size : stride, start_location_y-patch_size : start_location_y + patch_size : stride]
    measure = measure / measure.max()
    mask = mask[start_location_x-patch_size : start_location_x + patch_size : stride, start_location_y-patch_size : start_location_y + patch_size : stride, :]
    mask = mask/mask.max()
    mask = mask/10
    phi = mask
    opt.Phisum = np.sum(np.square(phi), axis=2)
    opt.Phisum = np.where(opt.Phisum == 0, 1, opt.Phisum)
    # phi = 
    gapDenoise = GapDenoise(opt)
    x = gapDenoise(mask, measure)
    logger.debug(x.shape)
    ImageUtils.show_hsi_line(x[patch_size-1, patch_size-1, :], "./line.png")

def test_load_mask():
    ImageUtils.load_mask_h5py("./mask/mask_0412_2048_2048_0_400_div_3.mat")

def show_true_hsi():
    hsi = ImageUtils.load_hsi_h5py("./image/real_hsi/org_sence_light_0416_16.h5")
    ImageUtils.show_hsi(ImageUtils.transpose_hsi(hsi), (24, 15, 4))

def show_mask():
    mask = ImageUtils.load_mask_h5py("./mask/mask_0412_2048_2048_0_400.mat")
    # logger.debug(mask.shape)
    ImageUtils.show_hsi(mask, (24, 15, 4))

if __name__=='__main__':
    test_gaptv_denoise()
    # show_mask()
    # show_true_hsi()
