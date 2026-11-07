
import sys
import os
import ctypes
import numpy as np
import cv2

from ctypes import *


# =====================================================
# 1. IMPORT HIKROBOT SDK
# =====================================================

MVS_PYTHON_PATH = (
    r"C:\Program Files (x86)\MVS"
    r"\Development\Samples\Python\MvImport"
)

if not os.path.exists(MVS_PYTHON_PATH):
    MVS_PYTHON_PATH = (
        r"C:\Program Files\MVS"
        r"\Development\Samples\Python\MvImport"
    )

if not os.path.exists(MVS_PYTHON_PATH):
    raise FileNotFoundError(
        "Cannot find Hikrobot MVS Python SDK. "
        "Please check MVS installation path."
    )

sys.path.insert(0, MVS_PYTHON_PATH)

print("SDK path:", MVS_PYTHON_PATH)
print("Folder exists:", os.path.exists(MVS_PYTHON_PATH))
print(
    "SDK file exists:",
    os.path.isfile(
        os.path.join(MVS_PYTHON_PATH, "MvCameraControl_class.py")
    )
)

from MvCameraControl_class import *


# =====================================================
# 2. HIKROBOT CAMERA CLASS
# =====================================================

class HikrobotCamera:

    def __init__(self, camera_index=0):

        self.camera_index = camera_index

        self.camera = None
        self.device_list = MV_CC_DEVICE_INFO_LIST()

        self.is_opened = False
        self.is_grabbing = False

        self.buffer = None
        self.buffer_size = 0

    # -------------------------------------------------
    # Enumerate devices
    # -------------------------------------------------

    def enumerate_devices(self):

        device_type = (
            MV_GIGE_DEVICE |
            MV_USB_DEVICE
        )

        ret = MvCamera.MV_CC_EnumDevices(
            device_type,
            self.device_list
        )

        if ret != 0:
            raise RuntimeError(
                f"Enumerate devices failed: 0x{ret:08x}"
            )

        device_count = self.device_list.nDeviceNum

        print(f"\nFound {device_count} camera(s)")

        if device_count == 0:
            return 0

        for i in range(device_count):

            device_info = self.device_list.pDeviceInfo[i].contents

            model_name = bytes(
                device_info.SpecialInfo.stGigEInfo.chModelName
            ).split(b'\0')[0].decode(
                errors="ignore"
            )

            serial_number = bytes(
                device_info.SpecialInfo.stGigEInfo.chSerialNumber
            ).split(b'\0')[0].decode(
                errors="ignore"
            )

            print(f"Camera {i}")
            print(f"  Model: {model_name}")
            print(f"  Serial: {serial_number}")

        return device_count

    # -------------------------------------------------
    # Open camera
    # -------------------------------------------------

    def open(self):

        if self.enumerate_devices() == 0:
            raise RuntimeError("No camera detected!")

        if self.camera_index >= self.device_list.nDeviceNum:
            raise IndexError("Invalid camera index")

        self.camera = MvCamera()

        # Get device information
        device_info = self.device_list.pDeviceInfo[
            self.camera_index
        ].contents

        # Create camera handle
        ret = self.camera.MV_CC_CreateHandle(device_info)

        if ret != 0:
            raise RuntimeError(
                f"Create handle failed: 0x{ret:08x}"
            )

        # Open camera
        ret = self.camera.MV_CC_OpenDevice()

        if ret != 0:
            raise RuntimeError(
                f"Open camera failed: 0x{ret:08x}"
            )

        self.is_opened = True

        print("\nCamera opened successfully!")

        # Optimize GigE packet size
        packet_size = self.camera.MV_CC_GetOptimalPacketSize()

        if packet_size > 0:

            ret = self.camera.MV_CC_SetIntValue(
                "GevSCPSPacketSize",
                packet_size
            )

            if ret != 0:
                print("Warning: Cannot set optimal packet size")

        # Disable trigger mode
        ret = self.camera.MV_CC_SetEnumValue(
            "TriggerMode",
            MV_TRIGGER_MODE_OFF
        )

        if ret != 0:
            raise RuntimeError(
                f"Set TriggerMode failed: 0x{ret:08x}"
            )

        # Get image payload size
        payload = MVCC_INTVALUE()

        memset(byref(payload), 0, sizeof(payload))

        ret = self.camera.MV_CC_GetIntValue(
            "PayloadSize",
            payload
        )

        if ret != 0:
            raise RuntimeError(
                f"Get PayloadSize failed: 0x{ret:08x}"
            )

        self.buffer_size = payload.nCurValue

        self.buffer = (c_ubyte * self.buffer_size)()

        print(f"Payload size: {self.buffer_size} bytes")

        return True

    # -------------------------------------------------
    # Start acquisition
    # -------------------------------------------------

    def start_grabbing(self):

        if not self.is_opened:
            raise RuntimeError("Camera is not opened")

        ret = self.camera.MV_CC_StartGrabbing()

        if ret != 0:
            raise RuntimeError(
                f"Start grabbing failed: 0x{ret:08x}"
            )

        self.is_grabbing = True

        print("Camera acquisition started!")

    # -------------------------------------------------
    # Get one frame
    # -------------------------------------------------

    def get_frame(self, timeout_ms=1000):

        if not self.is_grabbing:
            raise RuntimeError("Camera is not grabbing")

        frame_info = MV_FRAME_OUT_INFO_EX()

        memset(
            byref(frame_info),
            0,
            sizeof(frame_info)
        )

        ret = self.camera.MV_CC_GetOneFrameTimeout(
            self.buffer,
            self.buffer_size,
            frame_info,
            timeout_ms
        )

        if ret != 0:
            return None

        width = frame_info.nWidth
        height = frame_info.nHeight
        frame_length = frame_info.nFrameLen

        # Destination BGR buffer
        dst_size = width * height * 3

        dst_buffer = (c_ubyte * dst_size)()

        convert_param = MV_CC_PIXEL_CONVERT_PARAM()

        memset(
            byref(convert_param),
            0,
            sizeof(convert_param)
        )

        convert_param.nWidth = width
        convert_param.nHeight = height

        convert_param.pSrcData = cast(
            self.buffer,
            POINTER(c_ubyte)
        )

        convert_param.nSrcDataLen = frame_length

        convert_param.enSrcPixelType = (
            frame_info.enPixelType
        )

        convert_param.enDstPixelType = (
            PixelType_Gvsp_BGR8_Packed
        )

        convert_param.pDstBuffer = cast(
            dst_buffer,
            POINTER(c_ubyte)
        )

        convert_param.nDstBufferSize = dst_size

        ret = self.camera.MV_CC_ConvertPixelType(
            convert_param
        )

        if ret != 0:
            raise RuntimeError(
                f"Pixel conversion failed: 0x{ret:08x}"
            )

        # Convert to NumPy array
        image = np.frombuffer(
            dst_buffer,
            dtype=np.uint8,
            count=dst_size
        )

        image = image.reshape(
            height,
            width,
            3
        ).copy()

        return image

    # -------------------------------------------------
    # Stop acquisition
    # -------------------------------------------------

    def stop_grabbing(self):

        if self.is_grabbing:

            self.camera.MV_CC_StopGrabbing()

            self.is_grabbing = False

            print("Acquisition stopped!")

    # -------------------------------------------------
    # Close camera
    # -------------------------------------------------

    def close(self):

        if self.camera is not None:

            self.stop_grabbing()

            if self.is_opened:

                self.camera.MV_CC_CloseDevice()

                self.is_opened = False

            self.camera.MV_CC_DestroyHandle()

            self.camera = None

            print("Camera closed!")

    # -------------------------------------------------
    # Context manager
    # -------------------------------------------------

    def __enter__(self):

        self.open()
        self.start_grabbing()

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self.close()
