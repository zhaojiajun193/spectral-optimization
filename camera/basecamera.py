from abc import abstractmethod, ABCMeta
#抽象类

class baseCamera(metaclass=ABCMeta):
    
    @abstractmethod
    def get_one_image(self):
        pass

    @abstractmethod
    def get_exposure_time(self):
        pass

    @abstractmethod
    def set_exposure_time(self, exposure_time):
        pass

    @abstractmethod
    def stop_grabing(self):
        pass

    @abstractmethod
    def close_camera(self):
        pass

    @abstractmethod
    def start_grabing(self):
        pass

    @abstractmethod
    def get_frame_rate(self):
        pass

    @abstractmethod
    def set_frame_rate(self, frame_rate):
        pass

    @abstractmethod
    def get_pixel_merge(self, pixel_merge):
        pass

    @abstractmethod
    def set_pixel_merge(self, pixel_merge):
        pass
    