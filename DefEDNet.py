import torch
import torch.nn as nn
import torch.nn.functional as F
from .def_resnet import def_resnet34
from functools import partial
from .defconv import DefC

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

nonlinearity = partial(F.relu, inplace=True)


class Ladder_ASPP(nn.Module):
    def __init__(self, channel):
        super(Ladder_ASPP,self).__init__()
        self.dilate1 = SeparableConv2d(channel,channel,kernel_size=3,dilation=1,padding=1)
        self.dilate2 = SeparableConv2d(channel*2,channel,kernel_size=3,dilation=2,padding=2)
        self.dilate3 = SeparableConv2d(channel*3,channel,kernel_size=3,dilation=5,padding=5)
        self.dilate4 = SeparableConv2d(channel*4,channel,kernel_size=3,dilation=7,padding=7)
        self.bn = nn.BatchNorm2d(channel)
        self.drop = nn.Dropout2d(0.5)
        self.sg = nn.Sigmoid()
        
        self.finalchannel = channel

        self.conv1x1_1 = SeparableConv2d(channel*5, channel*3, kernel_size=1, dilation=1, padding=0)
        self.conv1x1_2 = SeparableConv2d(channel*3, channel*2, kernel_size=1, dilation=1, padding=0)

        # Master branch
        self.conv_master = SeparableConv2d(channel, channel, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channel)

        # Global pooling branch
        self.conv_gpb = SeparableConv2d(channel, channel, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        

        x_gpb = self.avg_pool(x)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)
        x_gpb = self.sg(x_gpb)

        x_se = x_gpb * x

        #first block rate1
        d1 = self.dilate1(x)
        d1 = self.bn(d1)

        #second block rate3
        d2 = torch.cat([d1,x],1)
        d2 = self.dilate2(d2)
        d2 = self.bn(d2)

        #third block rate5
        d3 = torch.cat([d1,d2,x],1)
        d3 = self.dilate3(d3)
        d3 = self.bn(d3)

        #last block rate7
        d4 = torch.cat([d1,d2,d3,x],1)
        d4 = self.dilate4(d4)
        d4 = self.bn(d4)
        
        out = torch.cat([d1,d2,d3,d4,x_se],1)
        out = self.drop(out)
        out = self.conv1x1_1(out)
        out = self.conv1x1_2(out)

        
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = SeparableConv2d(in_channels,in_channels//4,kernel_size=3,stride=1,padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = SeparableConv2d(in_channels // 4, n_filters,kernel_size=3,stride=1,padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DefED_Net(nn.Module):
    def __init__(self, num_classes=3):
        super(DefED_Net, self).__init__()

        filters = [64, 128, 256, 512,1024]
        
        def_resnet = def_resnet34()
        
        self.firstconv = DefC(1,64,7,stride=2,padding=3,bias=False)
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(inplace=True)
        self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        self.sconv = DefC(64,64,3,2,1)


        self.xe4 = DefC(64,64,3,4)
        self.xe3 = DefC(64,64,3,2,1)
        self.xe2 = DefC(64,64,3,1,1)

        self.encoder1 = def_resnet.layer1
        self.encoder2 = def_resnet.layer2
        self.encoder3 = def_resnet.layer3
        self.encoder4 = def_resnet.layer4


        self.ladder_aspp = Ladder_ASPP(512)

        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(576, 256)
        self.decoder2 = DecoderBlock(320, 128)
        self.decoder1 = DecoderBlock(192, 64)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = SeparableConv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = SeparableConv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_p = self.sconv(x)
        x_p = self.drop(x_p)

        xe_4 = self.xe4(x_p)
        xe_3 = self.xe3(x_p)
        xe_2 = self.xe2(x_p)
        
        e1 = self.encoder1(x_p)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        e4 = self.drop(e4)

        # Center
        e4 = self.ladder_aspp(e4)


        # Decoder
        d4 = self.decoder4(e4)
        d4 = self.drop(d4)
        d3 = self.decoder3(torch.cat([d4,xe_4],1))
        d3 = self.drop(d3)
        d2 = self.decoder2(torch.cat([d3,xe_3],1))
        d2 = self.drop(d2)
        d1 = self.decoder1(torch.cat([d2,xe_2],1))
        d1 = self.drop(d1)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)