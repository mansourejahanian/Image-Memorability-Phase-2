import torch
import torch.nn as nn
import torchvision.models as model

class MyResnet(nn.Module):

    def __init__(self):
        super(MyResnet, self).__init__()
        self.resnet = model.resnet18(pretrained=True)

    def _forward_impl(self, x):
        c1 = self.resnet.conv1(x)
        out = self.resnet.bn1(c1)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)

        layer1 = self.resnet.layer1(out)
        layer2 = self.resnet.layer2(layer1)

        layer3 = self.resnet.layer3(layer2)

        layer4 = self.resnet.layer4(layer3)

        out = self.resnet.avgpool(layer4)
        out = torch.flatten(out, start_dim=1)
        fc1 = self.resnet.fc(out)

        return [c1, layer1, layer2, layer3, layer4, fc1[:, :, None, None]]

    def forward(self, x):
        return self._forward_impl(x)

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
        self.extractor = MyResnet()
        # Remove the fully connected layer at the end
        # self.resnet = MyResnet()
        # self.extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, _x):
        return self.extractor((_x - self.mean[None, :, None, None])/self.std[None, :, None, None])
    