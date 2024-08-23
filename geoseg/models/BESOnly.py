import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# 不直接把三个特征一次性融合
# 100 861
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 看之后可不可以改成bes中的上采样
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ConvBnReLU(nn.Sequential):
    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch))

        if relu:
            self.add_module("relu", nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BE_Module(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch4, mid_ch, out_ch, n_class):
        super(BE_Module, self).__init__()

        self.convb_1 = ConvBnReLU(in_ch1, mid_ch, kernel_size=1, stride=2, padding=0, dilation=1)  # 用来统一channel
        self.convb_2 = ConvBnReLU(in_ch2, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convb_5 = ConvBnReLU(in_ch4, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convbloss = nn.Conv2d(mid_ch, n_class, kernel_size=1, bias=False)
        boundary_ch = 3 * mid_ch
        self.boundaryconv = ConvBnReLU(boundary_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

    def forward(self, l1, l2, l5):
        # 统一channel,F5需要上采
        l1_b = self.convb_1(l1)
        l2_b = self.convb_2(l2)
        l5_b = self.convb_5(l5)
        l5_b = F.interpolate(l5_b, l2.size()[2:], mode='bilinear', align_corners=True)

        l1_bl = self.convbloss(l1_b)  # l1的boundary
        l2_bl = self.convbloss(l2_b)

        l5_bl = self.convbloss(l5_b)

        b = torch.cat((l1_b, l2_b, l5_b), dim=1)  # 输出的boundary feature Fb
        b = self.boundaryconv(b)

        c_boundaryloss = l1_bl + l2_bl + l5_bl

        return b, c_boundaryloss


class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels, inter_channels, out_features_num=4, attention_type='scale_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels//4, out_features_num)

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, concat_x, features_list):
        # N×C×H×W
        concat_x = self.conv(concat_x)  # 3×3卷积生成中间feature
        # C×H×W
        score = self.enhanced_attention(concat_x)  # 算attention weight

        assert len(features_list) == self.out_features_num
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = F.interpolate(score, size=shape, mode='bilinear')
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i+1] * features_list[i])  # feature × 对应的weight
        return torch.cat(x, dim=1)

# 可以换成其他attention
class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.Sequential(
            # Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x


class PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, nn.BatchNorm2d)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)



class Fusion_Module(nn.Module):
    def __init__(self, x_in, b_in, f_in):
        super(Fusion_Module, self).__init__()

        self.conv1 = ConvBnReLU(x_in, b_in, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2 = ConvBnReLU(b_in, f_in, kernel_size=1, stride=1, padding=0, dilation=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, ff):

        x_guide = F.interpolate(x, ff.size()[2:], mode='bilinear', align_corners=True)
        x_guide = self.conv2(x_guide)
        ff_guide = torch.mul(x_guide, ff)
        # fe = ff + ff_guide + f5_guide

        return ff_guide+ff


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv5(x)


class PoolUNet(nn.Module):
    def __init__(self, n_classes, backbone='resnet18', bilinear=False):
        super(PoolUNet, self).__init__()

        self.backbone = timm.create_model(backbone, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)

        self.down = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        self.n_classes = n_classes
        self.bilinear = bilinear
        # m_out_sz = 512
        in_channels = [64, 128, 256, 512]
        inner_channels = 256
        self.asf = ScaleFeatureSelection(256, inner_channels // 4)
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1)
        m_out_sz = 512
        # self.ppm = PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6])
        self.ppm = nn.Sequential(PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6]),
                                 nn.Conv2d(m_out_sz // 4, n_classes, kernel_size=1))
        self.fusion = Fusion_Module(256, 128, 64)
        self.d = nn.Conv2d(256, 64, kernel_size=1)
        self.be = BE_Module(64, 128, 512, 64, 128, 1)
        self.up1 = Up(512, 256, bilinear)
        self.uni1 = nn.Conv2d(262, 256, kernel_size=1, bias=False)
        self.up2 = Up(256, 128, bilinear)
        self.uni2 = nn.Conv2d(134, 128, kernel_size=1, bias=False)
        self.up3 = Up(128, 64, bilinear)
        self.uni3 = nn.Conv2d(70, 64, kernel_size=1, bias=False)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.outc = FCNHead(32, n_classes)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.backbone(x)
        csize = [c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]]
        # 先把fuse的channel降低 之后可以试试在生成fuse时就降低channel

        be, c_boundaryloss = self.be(c1, c2, c4)
        x = self.up1(c4, c3)
        x = self.up2(x, be)
        a = self.up3(x, c1)
        x = self.up4(a)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        if self.training:
            return x, c_boundaryloss
        return x