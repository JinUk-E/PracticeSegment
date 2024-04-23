from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.linalg import svd
from numpy.random import normal
from math import sqrt

class UNet(nn.Module):
    def __init__(self,colorDim=1):
        super(UNet, self).__init__()
        self.conv1_1 = nn.Conv2d(colorDim, 64, 3)  # input of (n,n,1), output of (n-2,n-2,64)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 1024, 3)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3)
        self.upconv5 = nn.Conv2d(1024, 512, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn5_out = nn.BatchNorm2d(1024)
        self.conv6_1 = nn.Conv2d(1024, 512, 3)
        self.conv6_2 = nn.Conv2d(512, 512, 3)
        self.upconv6 = nn.Conv2d(512, 256, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn6_out = nn.BatchNorm2d(512)
        self.conv7_1 = nn.Conv2d(512, 256, 3)
        self.conv7_2 = nn.Conv2d(256, 256, 3)
        self.upconv7 = nn.Conv2d(256, 128, 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn7_out = nn.BatchNorm2d(256)
        self.conv8_1 = nn.Conv2d(256, 128, 3)
        self.conv8_2 = nn.Conv2d(128, 128, 3)
        self.upconv8 = nn.Conv2d(128, 64, 1)
        self.bn8 = nn.BatchNorm2d(64)
        self.bn8_out = nn.BatchNorm2d(128)
        self.conv9_1 = nn.Conv2d(128, 64, 3)
        self.conv9_2 = nn.Conv2d(64, 64, 3)
        self.conv9_3 = nn.Conv2d(64, colorDim, 1)
        self.bn9 = nn.BatchNorm2d(colorDim)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(x)))))
        x2 = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(self.maxpool(x1))))))
        x3 = F.relu(self.bn3(self.conv3_2(F.relu(self.conv3_1(self.maxpool(x2))))))
        x4 = F.relu(self.bn4(self.conv4_2(F.relu(self.conv4_1(self.maxpool(x3))))))
        xup = F.relu(self.conv5_2(F.relu(self.conv5_1(self.maxpool(x4)))))

        xup = self.bn5(self.upconv5(self.upsample(xup)))
        cropIdx = (x4.size(2) - xup.size(2)) // 2
        x4 = x4[:, :, cropIdx:cropIdx + xup.size(2), cropIdx:cropIdx + xup.size(2)]

        xup = self.bn5_out(torch.cat((x4, xup), 1))
        xup = F.relu(self.conv6_2(F.relu(self.conv6_1(xup))))

        xup = self.bn6(self.upconv6(self.upsample(xup)))
        cropIdx = (x3.size(2) - xup.size(2)) // 2
        x3 = x3[:, :, cropIdx:cropIdx + xup.size(2), cropIdx:cropIdx + xup.size(2)]

        xup = self.bn6_out(torch.cat((x3, xup), 1))
        xup = F.relu(self.conv7_2(F.relu(self.conv7_1(xup))))

        xup = self.bn7(self.upconv7(self.upsample(xup)))
        cropIdx = (x2.size(2) - xup.size(2)) // 2
        x2 = x2[:, :, cropIdx:cropIdx + xup.size(2), cropIdx:cropIdx + xup.size(2)]

        xup = self.bn7_out(torch.cat((x2, xup), 1))
        xup = F.relu(self.conv8_2(F.relu(self.conv8_1(xup))))

        xup = self.bn8(self.upconv8(self.upsample(xup)))
        cropIdx = (x1.size(2) - xup.size(2)) // 2
        x1 = x1[:, :, cropIdx:cropIdx + xup.size(2), cropIdx:cropIdx + xup.size(2)]

        xup = self.bn8_out(torch.cat((x1, xup), 1))
        xup = F.relu(self.conv9_3(F.relu(self.conv9_2(F.relu(self.conv9_1(xup))))))

        return F.softsign(self.bn9(xup))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

unet = UNet().cuda()