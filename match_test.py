import cv2
from utils import ImageUtils
from loguru import logger

def image_match(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    logger.debug(result.shape)
    result = ImageUtils.resize_image(result, result.shape[1]/4, result.shape[0]/4)
    logger.debug(result.shape)
    cv2.imshow("orb-match", result)
    cv2.waitKey(0)

if __name__=='__main__':
    img1_path = "./1.png"
    img2_path = "./image/0416_1.bmp"
    img1 = ImageUtils.load_color_image(img1_path)

    logger.debug(img1.size)
    img2 = ImageUtils.load_mono_image(img2_path)
    img1 = ImageUtils.cvt_color_2_grayscale(img1)

    image_match(img1, img2)