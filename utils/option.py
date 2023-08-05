class Opt:
    def __init__(self):
        self.opt_dict = {}

    @property
    def rate(self):
        return self.opt_dict['rate']

    #使用property和getter、setter管理配置
    @rate.setter
    def rate(self, value):
        self.opt_dict['rate'] = value
        
    @property
    def acc(self):
        return self.opt_dict['acc']

    @acc.setter
    def acc(self, value:bool):
        #acc指的是是否启用加速
        self.opt_dict['acc'] = value

    @property
    def flag_iqa(self):
        return self.opt_dict['flag_iqa']

    @flag_iqa.setter
    def flag_iqa(self, value:bool):
        #是否使用图像质量评价
        self.opt_dict['flag_iqa'] = value

    @property
    def nframe(self):
        return self.opt_dict['nframe']

    @nframe.setter
    def nframe(self, value):
        #有几张照片需要重建
        self.opt_dict['nframe'] = value

    @property
    def MAXB(self):
        return self.opt_dict['MAXB']

    @MAXB.setter
    def MAXB(self, value):
        #重建出来的最大值
        self.opt_dict['MAXB'] = value

    @property
    def denoiser(self):
        return self.opt_dict['denoiser']

    @denoiser.setter
    def denoiser(self, value:str):
        #去噪方法 选用tv
        self.opt_dict['denoiser'] = value

    @property
    def iter(self):
        return self.opt_dict['iter']

    @iter.setter
    def iter(self, value):
        #迭代次数
        self.opt_dict['iter'] = value
    
    @property
    def tvweight(self):
        #tv的权重
        return self.opt_dict['tvweight']

    @tvweight.setter
    def tvweight(self, value):
        self.opt_dict['tvweight'] = value

    @property
    def tviter(self):
        #tv迭代次数
        return self.opt_dict['tviter']

    @tviter.setter
    def tviter(self, value):
        #tv迭代次数
        self.opt_dict['tviter'] = value
    
    @property
    def Phisum(self):
        return self.opt_dict['Phisum']

    @Phisum.setter
    def Phisum(self, value):
        self.opt_dict['Phisum'] = value
