from camera import baseCamera
from camera.MvImport.MvCameraControl_class import *
from loguru import logger
import sys
import msvcrt
from ctypes import *
import numpy as np
from utils import ImageUtils
import cv2
from time import sleep
#代码思路，仿照CameraOperation类实现
#init实现
#openDevice
#closeDevice
#get_parameter
#set_parameter
#get_image
#save_image 
#start_grabing
#close_grabing
#存储图像和发送图像写到start_grabing启动的线程中去
#get_one_image 取一张图像
class HikCamera(baseCamera):

    def __init__(self, st_device_list, userDefinedName):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            logger.error("enum devices fail! ret[0x%x]" % ret)
            sys.exit()
        if deviceList.nDeviceNum == 0:
            logger.debug("find no device!")
            sys.exit()
        logger.debug("Find %d devices!" % deviceList.nDeviceNum)
        if deviceList.nDeviceNum > 1:
            logger.error("Find more than one device, please just connect one device")
            sys.exit()
        mvcc_dev_info = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            logger.debug("\ngige device: [%d]" % 0)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            logger.debug("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            logger.debug("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            logger.debug("\nu3v device: [%d]" % 0)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            logger.debug("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            logger.debug("user serial number: %s" % strSerialNumber)
        self.cam = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            logger.error("create handle fail! ret[0x%x]" % ret)
            sys.exit()
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            logger.error("open device fail! ret[0x%x]" % ret)
            sys.exit()
        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                if ret != 0:
                    logger.error("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                logger.error("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        stBool = c_bool(False)
        ret = self.cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
        if ret != 0:
            logger.error("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)

        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            logger.error("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            logger.error("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()
        logger.debug("海康相机初始化成功")

    def get_one_image(self):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            logger.debug("get one frame: Width[%d], Height[%d], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
            pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
            cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
            data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),dtype=np.uint8)
            image = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
            nRet = self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            return image
        else:
            logger.error("no data[0x%x]" % ret)
        nRet = self.cam.MV_CC_FreeImageBuffer(stOutFrame)
        return None

    def get_exposure_time(self):
        stFloatParam_ExposureTime = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_ExposureTime)
        if ret != 0:
            logger.error("get exposureTime fail! ret[0x%x]" % ret)
            return False, 0
        return True, stFloatParam_ExposureTime.fCurValue
    
    def set_exposure_time(self, exposure_time):
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret != 0:
            logger.error("set exposureTime fail! ret[0x%x]" % ret)
            return False
        return True

    def stop_grabing(self):
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            logger.error("stop grabbing fail! ret[0x%x]" % ret)
            sys.exit(1)

    def close_camera(self):
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            logger.error("close deivce fail! ret[0x%x]" % ret)
            sys.exit()
        logger.debug("close device success!")
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            logger.error("destroy handle fail! ret[0x%x]" % ret)
            sys.exit()
        logger.debug("destroy handle success!")

    def auto_exposure_adjustment(self, mean_min, mean_max):
        while True:
            try:
                image = self.get_one_image()
                # cv2.imshow("test", image)
                image_mean = ImageUtils.get_monoimage_histmean(image)
                if image_mean >= 50 and image_mean <= 100:
                    logger.debug(image_mean)
                    break
                else:
                    gain = 75 / image_mean
                    logger.debug(gain)
                    if gain > 10:
                        gain = 10
                    ret, exposure_time = self.get_exposure_time()
                    want_exposure_time = gain*exposure_time
                    logger.debug(want_exposure_time)
                    if want_exposure_time > 1000000:
                        want_exposure_time = 1000000
                    ret = self.set_exposure_time(want_exposure_time)
                    sleep(1)
            except Exception as e:
                logger.debug(e)
        logger.info("adjust success")