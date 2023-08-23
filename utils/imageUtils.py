import cv2
import copy
import numpy as np
import spectral
from loguru import logger
from scipy.io import loadmat, savemat
import h5py
import matplotlib.pyplot as plt

class ImageUtils:
    
    @staticmethod
    def circle_point(image, x, y):
        image = copy.deepcopy(image)
        return cv2.circle(img=image, center=(x, y), radius=10, color=(255, 0, 0), thickness=-1)

    @staticmethod
    def resize_image(image, width, height):
        # logger.debug(width)
        # logger.debug(height)
        image = cv2.resize(image, (int(width), int(height)))
        return image

    @staticmethod
    def load_mono_image(path):
        image = cv2.imread(path, -1)
        return image

    @staticmethod
    def load_color_image(path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def transpose_hsi(hsi):
        hsi = copy.deepcopy(hsi)
        return hsi.transpose(1, 2, 0)

    @staticmethod
    def cvt_color_rgb_2_bgr(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def cvt_color_bgr_2_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def cvt_color_2_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def save_rgb_image(save_path, image):
        cv2.imwrite(save_path, image)

    @staticmethod
    def show_hsi_line(hsi_line, save_path):
        fig, ax = plt.subplots()
        channels = hsi_line.shape[0]
        ax.plot(np.arange(1, 62), hsi_line, c='b')
        logger.debug(hsi_line.shape)
        fig.savefig(save_path)

    @staticmethod
    #bands = (24, 15, 4)
    #hsi -> (2048, 2048, 61)
    def show_hsi(hsi, bands):
        spectral.imshow(hsi, bands)
        rgb = hsi[:, :, bands]
        rgb = rgb/rgb.max()
        rgb = np.maximum(rgb, 0)
        rgb = (rgb * 255).astype(np.uint8)
        # logger.debug(rgb.shape)
        # logger.debug(rgb.shape[0])
        # logger.debug(rgb.shape[1])
        rgb = ImageUtils.resize_image(rgb, rgb.shape[0]/4, rgb.shape[1]/4)
        rgb = ImageUtils.cvt_color_rgb_2_bgr(rgb)
        cv2.imshow("rgb", rgb)
        cv2.waitKey(0)

    @staticmethod
    def save_hsi(hsi, save_path):
        savemat(save_path, {'hsi':hsi})

    @staticmethod
    def load_hsi_scio(file_path):
        data = loadmat(file_path)
        return data['hsi']

    @staticmethod
    def load_hsi_h5py(file_path):
        f = h5py.File(file_path, 'r')
        logger.debug(f.keys())
        hsi = f['hsi'][:]
        # hsi = np.transpose(hsi, (1, 0, 2))
        return hsi

    @staticmethod
    def load_mask_h5py(file_path):
        f = h5py.File(file_path, 'r')
        mask = f['mask'][:]
        #(2048, 2048, 61)
        logger.debug(mask.shape)
        mask = np.transpose(mask, (1, 0, 2))
        return mask

    @staticmethod
    def load_mask_scio(file_path):
        data = loadmat(file_path)
        logger.debug(data.shape)

    @staticmethod
    #b
    def HSI2RGB_function(bands, hsi):
        CIE1931 = np.array([[380, 0.0272, -0.0115, 0.9843],
        [385, 0.0268, -0.0114, 0.9846],
        [390, 0.0263, -0.0114, 0.9851],
        [395, 0.0256, -0.0113, 0.9857],
        [400, 0.0247, -0.0112, 0.9865],
        [405, 0.0237, -0.0111, 0.9874],
        [410, 0.0225, -0.0109, 0.9884],
        [415, 0.0207, -0.0104, 0.9897],
        [420, 0.0181, -0.0094, 0.9913],
        [425, 0.0142, -0.0076, 0.9934],
        [430, 0.0088, -0.0048, 0.9960],
        [435, 0.0012, -0.0007, 0.9995],
        [440, -0.0084, 0.0018, 1.0036],
        [445, -0.0213, 0.0120, 1.0093],
        [450, -0.0390, 0.0218, 1.0172],
        [455, -0.0618, 0.0345, 1.0273],
        [460, -0.0909, 0.0517, 1.0392],
        [465, -0.1281, 0.0762, 1.0519],
        [470, -0.1821, 0.1175, 1.0646],
        [475, -0.2584, 0.1840, 1.0744],
        [480, -0.3667, 0.2906, 1.0761],
        [485, -0.5200, 0.4568, 1.0632],
        [490, -0.7150, 0.6996, 1.0154],
        [495, -0.9459, 1.0247, 0.9212],
        [500, -1.1685, 1.3905, 0.7780],
        [505, -1.3182, 1.7195, 0.5987],
        [510, -1.3371, 1.9318, 0.4053],
        [515, -1.2076, 1.9699, 0.2377],
        [520, -0.9830, 1.8534, 0.1296],
        [525, -0.7386, 1.6662, 0.0724],
        [530, -0.5159, 1.4761, 0.0398],
        [535, -0.3304, 1.3105, 0.0199],
        [540, -0.1707, 1.1628, 0.0079],
        [545, -0.0293, 1.0282, 0.0011],
        [550, 0.0974, 0.9051, -0.0025],
        [555, 0.2121, 0.7919, -0.0040],
        [560, 0.3164, 0.6881, -0.0045],
        [565, 0.4112, 0.5932, -0.0044],
        [570, 0.4973, 0.5067, -0.0040],
        [575, 0.5751, 0.4283, -0.0034],
        [580, 0.6449, 0.3579, -0.0028],
        [585, 0.7071, 0.2952, -0.0023],
        [590, 0.7617, 0.2402, -0.0019],
        [595, 0.8087, 0.1928, -0.0015],
        [600, 0.8475, 0.1537, -0.0012],
        [605, 0.8800, 0.1209, -0.0009],
        [610, 0.9059, 0.0949, -0.0008],
        [615, 0.9265, 0.0741, -0.0006],
        [620, 0.9425, 0.0580, -0.0005],
        [625, 0.9550, 0.0454, -0.0004],
        [630, 0.9649, 0.0354, -0.0003],
        [635, 0.9730, 0.0272, -0.0002],
        [640, 0.9797, 0.0205, -0.0002],
        [645, 0.9850, 0.0152, -0.0002],
        [650, 0.9888, 0.0113, -0.0001],
        [655, 0.9918, 0.0083, -0.0001],
        [660, 0.9940, 0.0061, -0.0001],
        [665, 0.9954, 0.0047, -0.0001],
        [670, 0.9966, 0.0035, -0.0001],
        [675, 0.9975, 0.0025, 0.0000],
        [680, 0.9984, 0.0016, 0.0000],
        [685, 0.9991, 0.0009, 0.0000],
        [690, 0.9996, 0.0004, 0.0000],
        [695, 0.9999, 0.0001, 0.0000],
        [700, 1.0000, 0.0000, 0.0000],
        [705, 1.0000, 0.0000, 0.0000],
        [710, 1.0000, 0.0000, 0.0000],
        [715, 1.0000, 0.0000, 0.0000],
        [720, 1.0000, 0.0000, 0.0000],
        [725, 1.0000, 0.0000, 0.0000],
        [730, 1.0000, 0.0000, 0.0000],
        [735, 1.0000, 0.0000, 0.0000],
        [740, 1.0000, 0.0000, 0.0000],
        [745, 1.0000, 0.0000, 0.0000],
        [750, 1.0000, 0.0000, 0.0000],
        [755, 1.0000, 0.0000, 0.0000],
        [760, 1.0000, 0.0000, 0.0000],
        [765, 1.0000, 0.0000, 0.0000],
        [770, 1.0000, 0.0000, 0.0000],
        [775, 1.0000, 0.0000, 0.0000],
        [780, 1.0000, 0.0000, 0.0000]])

        select_index = []
        for i in range(len(bands)):
            index = np.where(CIE1931[:, 0]==bands[i])[0]
            select_index.append(index[0])
        select_index = np.array(select_index)
        select_cie = CIE1931[select_index, 1:]
        hsi_rgb = hsi[:len(bands), :, :]
        rgb = hsi_rgb.transpose(1, 2, 0) @ select_cie
        rgb = rgb / rgb.max()
        rgb = np.maximum(rgb, 0)
        # print('rgb:', rgb.shape, rgb.max(), rgb.mean(), rgb.min())
        # plt.imshow(rgb)
        # plt.show()
        return rgb