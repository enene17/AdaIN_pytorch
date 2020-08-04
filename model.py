import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from loss import mean_std
from loss import StyleLoss


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

cfgs = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):#):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding = 1, padding_mode = "reflection")   
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(arch, cfg, pretrained, progress, **kwargs):
    model = VGG(make_layers(cfgs[cfg]), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
        encoder_1 = nn.Sequential(*list(model.children())[0][:2])
        encoder_2 = nn.Sequential(*list(model.children())[0][2:7])
        encoder_3 = nn.Sequential(*list(model.children())[0][7:12])
        encoder_4 = nn.Sequential(*list(model.children())[0][12:21])
    return encoder_1, encoder_2, encoder_3, encoder_4

def vgg19(pretrained=True, progress=True, **kwargs):
    return _vgg('vgg19', 'E', pretrained, progress, **kwargs)


class DecoderNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(512, 256, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.relu3 = nn.ReLU(inplace=True)
        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.relu4 = nn.ReLU(inplace=True)
        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(256, 128, 3)
        self.relu5 = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.relu6 = nn.ReLU(inplace=True)
        self.pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(128, 64, 3)
        self.relu7 = nn.ReLU(inplace=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(64, 64, 3)
        self.relu8 = nn.ReLU(inplace=True)
        self.pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(64, 3, 3)      

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.upsample1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pad3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pad4(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pad5(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.upsample2(x)
        x = self.pad6(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pad7(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.upsample3(x)
        x = self.pad8(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.pad9(x)
        x = self.conv9(x)

        return x
        

class StyleTransferNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4 = vgg19()

        for i in range(4):
            for param in getattr(self, 'encoder_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

        self.decoder = DecoderNet()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def vgg_norm(self, x, batch_size):
        out = (x + 1) / 2 # [-1,1] -> [0,1]
        for i in range (0, batch_size):
            out[i] = self.normalize(out[i])
        return out

    def adain(self, c_feature, s_feature):
        size = c_feature.size()

        c_mean, c_std = mean_std(c_feature)
        s_mean, s_std = mean_std(s_feature)

        c_mean = c_mean.expand(size)
        c_std = c_std.expand(size)
        s_mean = s_mean.expand(size)
        s_std = s_std.expand(size)

        return s_std * (c_feature - c_mean) / c_std + s_mean
       
    def forward(self, c_img, s_img, a, batch_size):
        c_img_temp = self.vgg_norm(c_img, batch_size)
        s_img_temp = self.vgg_norm(s_img, batch_size)

        c_feature = self.encoder_1(c_img_temp)
        c_feature = self.encoder_2(c_feature)
        c_feature = self.encoder_3(c_feature)
        c_feature = self.encoder_4(c_feature)

        s_feature_1 = self.encoder_1(s_img_temp)
        s_feature_2 = self.encoder_2(s_feature_1)
        s_feature_3 = self.encoder_3(s_feature_2)
        s_feature_4 = self.encoder_4(s_feature_3)

        a_feature = self.adain(c_feature, s_feature_4)

        t = (1-a) * c_feature + a * a_feature

        result = self.decoder(t)

        result_temp = self.vgg_norm(result, batch_size)

        r_feature_1 = self.encoder_1(result_temp)
        r_feature_2 = self.encoder_2(r_feature_1)
        r_feature_3 = self.encoder_3(r_feature_2)
        r_feature_4 = self.encoder_4(r_feature_3)

        s_loss_1 = StyleLoss(s_feature_1, r_feature_1)
        s_loss_2 = StyleLoss(s_feature_2, r_feature_2)
        s_loss_3 = StyleLoss(s_feature_3, r_feature_3)
        s_loss_4 = StyleLoss(s_feature_4, r_feature_4)

        s_loss = s_loss_1 + s_loss_2 + s_loss_3 + s_loss_4

        return a_feature, r_feature_4, result, s_loss