import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable

"""
pre-trained ResNet
"""

class ResNet(nn.Module):
    """
    Args:
        fea_type: string, resnet101 or resnet 152
    """

    RESNET_TYPES=['resnet101', 'resnet152']


    def __init__(self, fea_type = 'resnet152'):
        super(ResNet, self).__init__()
        assert fea_type in self.RESNET_TYPES, "No such resnet." 
        self.fea_type = fea_type
        # rescale and normalize transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        resnet = self._get_resnet() 
        resnet.float()
        resnet.cuda()
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[: -3])
        self.pool5 = module_list[-2]

    
    def _get_resnet(self):
        resnet_fun = {
            'resnet101': lambda: models.resnet101(pretrained=True),
            'resnet152': lambda: models.resnet152(pretrained=True),
        }
        return resnet_fun[self.fea_type]()


    # rescale and normalize image, then pass it through ResNet
    def forward(self, x):
        x = self.transform(x)
        x = x.unsqueeze(0)  # reshape the single image s.t. it has a batch dim
        x = Variable(x).cuda()
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)
        return res_pool5