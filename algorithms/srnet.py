from .architecture import model_generator
import torch
import numpy as np

class SRnetDenoise:
    def __init__(self, method, pretrained_model_path):
        self._model = model_generator(method, pretrained_model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        self._start_x = 0
        self._start_y = 400
        pass
        #加载神经网络
    
    @property
    def model(self):
        return self._model

    @property
    def start_x(self):
        return self._start_x
    
    @property
    def start_y(self):
        return self._start_y

    def inference(self, measure):
        measure = measure[self.start_x:, self.start_y:]
        measure = measure / measure.max()
        measure = measure.astype(np.float32)
        measure = torch.from_numpy(measure)
        self.model.eval()
        measure = measure.unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = self.model(measure)
            outputs = outputs / outputs.max()
            outputs = torch.maximum(outputs, torch.tensor(0))
            outputs = outputs.squeeze().cpu().numpy()
            print('outputs', outputs.shape, outputs.max(), outputs.mean(), outputs.min())
        return outputs

