import torch
from .SRNet import SRNet
def model_generator(method, pretrained_model_path=None):

    if method == 'mst_plus_plus':
        model = MST_Plus_Plus().cuda()

    elif method == 'srnet':
        model = SRNet(in_channels=1, out_channels=61, dim=26, deep_stage=3, num_blocks=[1, 1, 1, 1], num_heads=[1, 2, 4, 8]).cuda()

    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model