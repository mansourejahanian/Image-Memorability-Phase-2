import torch
import torch.nn as nn
import torchvision.models as model


class Resnet_fmaps(nn.Module):
    '''
    image input dtype: float in range [0,1], size: 224, but flexible
    info on the dataloader compliant with the model database
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
    '''
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Resnet_fmaps, self).__init__()
        self.mean = nn.Parameter(torch.as_tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.as_tensor(std), requires_grad=False)
        self.extractor = model.resnet18(pretrained=True)

        # Remove the fully connected layer at the end
        self.extractor = nn.Sequential(*list(self.extractor.children())[:-1])


    def forward(self, _x):
        return self.extractor((_x - self.mean[None, :, None, None])/self.std[None, :, None, None])
    