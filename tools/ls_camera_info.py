'''
修改海康相机的DeviceUserID——用户自定义名称
'''
import sys
from MvImport.MvCameraControl_class import *


if __name__=='__main__':
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    
    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print ("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print ("find no device!")
        sys.exit()

    print ("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print ("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s", strSerialNumber)

            strUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                strUserDefinedName = strUserDefinedName + chr(per)
            print("user defined name: %s", strUserDefinedName)

        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print ("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)

            strUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                if per == 0:
                    break
                strUserDefinedName = strUserDefinedName + chr(per)
            print("user defined name: %s" % strUserDefinedName)