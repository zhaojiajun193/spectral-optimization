from camera import HikCamera
import sys
from loguru import logger
from utils import ImageUtils
from camera.MvImport.MvCameraControl_class import *
import cv2

def main():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0:
        logger.debug("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        logger.debug("Find No device!")
        sys.exit()
    print("Find %d devices!" % deviceList.nDeviceNum)
    hikCamera = HikCamera(None, deviceList, "spectral")
    hikCamera.Open_device()
    ret = hikCamera.Start_grabing()
    logger.debug("[%0x]" % ret)
    # for i in range(10000):
    #     image, status = hikCamera.get_image()
    #     if image is None:
    #         continue
    #     else:
    #         image = ImageUtils.resize_image(image, 512, 612)
    #         cv2.imshow("test", image)
    #         cv2.waitKey(25)

    # ImageUtils.save_rgb_image("./test.bmp", image)
    ret = hikCamera.Set_exposureTime(7000)
    exposure_time, ret = hikCamera.Get_exposureTime()
    if exposure_time is not None:
        print(type(exposure_time))
        logger.debug(exposure_time)
    ret = hikCamera.Set_frameRate(12)
    frame_rate, ret = hikCamera.Get_frameRate()
    if frame_rate is not None:
        print(type(frame_rate))
        logger.debug(frame_rate)
    ret = hikCamera.Set_gain(10)
    gain, ret = hikCamera.Get_gain()
    if gain is not None:
        print(type(gain))
        logger.debug(gain)
    # logger.debug(image.shape)
    hikCamera.Stop_grabing()
    

if __name__=='__main__':
    main()