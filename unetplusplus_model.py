
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        n1 = 64
        filters = [n1, n1*2, n1*4, n1*8, n1*16]

        self.conv0_0 = ConvBlock(in_ch, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        self.conv0_1 = ConvBlock(filters[0]+filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1]+filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2]+filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3]+filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0]*2+filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2+filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2]*2+filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0]*3+filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1]*3+filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0]*4+filters[1], filters[0])

        self.maxpool = nn.MaxPool2d(2, 2)
        self.upsample = lambda x, size: F.interpolate(x, size=size, mode="bilinear", align_corners=True)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0, x0_0.shape[2:])], 1))

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0, x1_0.shape[2:])], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1, x0_0.shape[2:])], 1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0, x2_0.shape[2:])], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1, x1_0.shape[2:])], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2, x0_0.shape[2:])], 1))

        x4_0 = self.conv4_0(self.maxpool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0, x3_0.shape[2:])], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1, x2_0.shape[2:])], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2, x1_0.shape[2:])], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3, x0_0.shape[2:])], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)
