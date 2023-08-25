from camera import baseCamera
from camera.MvImport.MvCameraControl_class import *
from loguru import logger
import sys
import msvcrt
from ctypes import *
import numpy as np
from utils import ImageUtils, ThreadUtils
import cv2
from time import sleep
import threading
import random
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
#get_one_image 取一张图像aaa
class HikCamera():

    def __init__(self, obj_cam, st_device_list, userDefinedName, b_open_device=False, b_start_grabbing=False,
                h_thread_handle=None,
                b_thread_closed=False, st_frame_info=None, b_exit=False, b_save_bmp=False, b_save_jpg=False,
                buf_save_image=None,
                n_save_image_size=0, n_win_gui_id=0, frame_rate=0, exposure_time=0, gain=0):
        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.userDefinedName = userDefinedName
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_grab_image = None
        self.buf_grab_image_size = 0
        self.buf_save_image = buf_save_image
        self.n_save_image_size = n_save_image_size
        self.h_thread_handle = h_thread_handle
        self.b_thread_closed
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        self.buf_lock = threading.Lock()

    def Open_device(self):
        if not self.b_open_device:
            for i in range(0, self.st_device_list.nDeviceNum):
                mvcc_dev_info = cast(self.st_device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                strUserDefinedName = ""
                if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                    print ("\ngige device: [%d]" % i)
                    for per in mvcc_dev_info.SpecialInfo.chUserDefinedName:
                        strUserDefinedName = strUserDefinedName + chr(per)
                    logger.debug("user defined name: %s", strUserDefinedName)
                elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                    print ("\nu3v device: [%d]" % i)
                    for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                        if per == 0:
                            break
                        strUserDefinedName = strUserDefinedName + chr(per)
                    logger.debug("user defined name: %s" % strUserDefinedName)
                if strUserDefinedName == self.userDefinedName:
                    self.obj_cam = MvCamera()
                    stDeviceList = cast(self.st_device_list.pDeviceInfo[int(i)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
                    ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
                    if ret != 0:
                        self.obj_cam.MV_CC_DestroyHandle()
                        return ret
                    ret = self.obj_cam.MV_CC_OpenDevice()
                    if ret != 0:
                        return ret
                    print("open device successfully!")
                    self.b_open_device = True
                    self.b_thread_closed = False
                    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                        nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                        if int(nPacketSize) > 0:
                            ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                            if ret != 0:
                                print("warning: set packet size fail! ret[0x%x]" % ret)
                        else:
                            print("warning: set packet size fail! ret[0x%x]" % nPacketSize)

                    stBool = c_bool(False)
                    ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
                    if ret != 0:
                        print("get acquisition frame rate enable fail! ret[0x%x]" % ret)

                    # ch:设置触发模式为off | en:Set trigger mode as off
                    ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
                    if ret != 0:
                        print("set trigger mode fail! ret[0x%x]" % ret)
                    return MV_OK
            return MV_E_CALLORDER

    def Start_grabing(self):
        if not self.b_start_grabbing and self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                return ret
            self.b_start_grabbing = True
            logger.debug(self.userDefinedName + "start grabbing successfully!")
            try:
                thread_id = random.randint(1, 10000)
                self.h_thread_handle = threading.Thread(target=HikCamera.Work_thread, args=(self,))
                self.h_thread_handle.start()
                self.b_thread_closed = True
                return MV_OK
            finally:
                pass
        return MV_E_CALLORDER

    def Stop_grabing(self):
        if self.b_start_grabbing and self.b_open_device:
            # 退出线程
            if self.b_thread_closed:
                ThreadUtils.Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
                ret = self.obj_cam.MV_CC_StopGrabbing()
                if ret != 0:
                    return ret
                print("stop grabbing successfully!")
                self.b_start_grabbing = False
                self.b_exit = True
                return MV_OK
            else:
                return MV_E_CALLORDER

    def Work_thread(self):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        img_buff = None
        numArray = None

        stPayloadSize = MVCC_INTVALUE_EX()
        ret_temp = self.obj_cam.MV_CC_GetIntValueEx("PayloadSize", stPayloadSize)
        if ret_temp != MV_OK:
            return
        NeedBufSize = int(stPayloadSize.nCurValue)
        while True:
            if self.buf_grab_image_size < NeedBufSize:
                self.buf_grab_image = (c_ubyte * NeedBufSize)()
                self.buf_grab_image_size = NeedBufSize

            ret = self.obj_cam.MV_CC_GetOneFrameTimeout(self.buf_grab_image, self.buf_grab_image_size, stFrameInfo)

            # ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:
                # 拷贝图像和图像信息
                if self.buf_save_image is None:
                    self.buf_save_image = (c_ubyte * stFrameInfo.nFrameLen)()
                self.st_frame_info = stFrameInfo
                # logger.debug("get one frame: Width[%d], Height[%d], nFrameNum[%d]"
                #       % (self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                # 获取缓存锁
                self.buf_lock.acquire()
                cdll.msvcrt.memcpy(byref(self.buf_save_image), self.buf_grab_image, self.st_frame_info.nFrameLen)
                self.buf_lock.release()

                # logger.debug("get one frame: Width[%d], Height[%d], nFrameNum[%d]"
                #       % (self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                # 释放缓存
                # self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print("no data, ret = " + To_hex_str(ret))
                continue

            # 是否退出
            if self.b_exit:
                if img_buff is not None:
                    del img_buff
                if self.buf_save_image is not None:
                    del self.buf_save_image
                break

    def get_image(self):
        if 0 == self.buf_save_image:
            return
        self.buf_lock.acquire()
        if self.st_frame_info is None:
            self.buf_lock.release()
            logger.debug("st frame info is None")
            return None, MV_E_CALLORDER
        else:
            data = np.frombuffer(self.buf_save_image, count=int(self.st_frame_info.nWidth * self.st_frame_info.nHeight),dtype=np.uint8)
            image = data.reshape(self.st_frame_info.nHeight, self.st_frame_info.nWidth, -1)
            self.buf_lock.release()
        return image, MV_OK

    def Close_device(self):
        if self.b_open_device:
            # 退出线程
            if self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                return ret

        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit = True
        print("close device successfully!")
        return MV_OK

    def Set_exposureTime(self, exposure_time):
        if self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
            if ret != 0:
                logger.debug("set Exposure Time fail! ret[0x%x]" % ret)
                return ret
            return MV_OK

    def Get_exposureTime(self):
        if self.b_open_device:
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            if ret != 0:
                logger.debug("get exposure time fail! ret[0x%x]" % ret)
                return None, ret
            #stFloatParam_exposureTime.fCurValue的类型是float
            return stFloatParam_exposureTime.fCurValue, MV_OK

    def Set_gain(self, gain):
        if self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
            if ret != 0:
                logger.debug("set gain fail! ret[0x%x]" % ret)
                return ret
            return MV_OK

    def Get_gain(self):
        if self.b_open_device:
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            if ret != 0:
                logger.debug("get gain fail! ret[0x%x]" % ret)
                return None, ret
            return stFloatParam_gain.fCurValue, MV_OK

    def Set_frameRate(self, frame_rate):
        if self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frame_rate))
            if ret != 0:
                logger.debug("set AcquisitionFrameRate fail! ret[0x%x]" % ret)
                return ret
            return MV_OK

    def Get_frameRate(self):
        if self.b_open_device:
            stFloatParam_frameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_frameRate), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_frameRate)
            if ret != 0:
                logger.debug("get frameRate fail! ret[0x%x]" % ret)
                return None, ret
            return stFloatParam_frameRate.fCurValue, MV_OK

    def Set_binning_vertical(self, binning_num):
        if self.b_open_device:
            ret = self.obj_cam.MV_CC_SetEnumValue("BinningVertical", int(binning_num))
            if ret != 0:
                logger.debug("set binning vertical fail! ret[0x%x]" % ret)
                return ret
            return MV_OK

    def Set_binning_horizontal(self, binning_num):
        if self.b_open_device:
            ret = self.obj_cam.MV_CC_SetEnumValue("BinningHorizontal", int(binning_num))
            if ret != 0:
                logger.debug("set binning horizontal fail! ret[0x%x]" % ret)
                return ret
            return MV_OK