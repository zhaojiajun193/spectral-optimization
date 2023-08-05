from algorithms import SRnetDenoise

class SRnetDenoiseTest:
    def __init__(self, method, pretrained_model_path):
        self.srnetDenoise = SRnetDenoise(method, pretrained_model_path)

    def test_inference(self, measure):
        return self.srnetDenoise.inference(measure)
