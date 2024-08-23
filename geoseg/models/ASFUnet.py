import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# 下采 不加权上采 每次上采后加两次卷积 unet0+ASF 第三次上采没加高级特征
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
        self.in5 = nn.Conv2d(in_channels[3], inner_channels, 1)
        self.in4 = nn.Conv2d(in_channels[2], inner_channels, 1)
        self.in3 = nn.Conv2d(in_channels[1], inner_channels, 1)
        self.in2 = nn.Conv2d(in_channels[0], inner_channels, 1)
        self.asfup4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.asfup3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.asfup2 = nn.Upsample(scale_factor=2, mode='nearest')
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
        self.asf = ScaleFeatureSelection(256, 256 // 4)
        self.d = nn.Conv2d(256, 64, kernel_size=1)

        self.up1 = Up(512, 256, bilinear)
        self.uni1 = nn.Conv2d(262, 256, kernel_size=1, bias=False)
        self.up2 = Up(256, 128, bilinear)
        self.uni2 = nn.Conv2d(134, 128, kernel_size=1, bias=False)
        self.up3 = Up(128, 64, bilinear)
        self.uni3 = nn.Conv2d(70, 64, kernel_size=1, bias=False)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.backbone(x)
        csize = [c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]]
        fp = self.ppm(c4)
        in4 = self.in5(c4)  # 统一channel
        in3 = self.in4(c3)
        in2 = self.in3(c2)
        in1 = self.in2(c1)
        out3 = self.asfup4(in4) + in3  # 1/16
        out2 = self.asfup3(out3) + in2  # 1/8
        out1 = self.asfup2(out2) + in1  # 1/4

        p4 = self.out5(in4)
        p3 = self.out4(out3)
        p2 = self.out3(out2)
        p1 = self.out2(out1)
        fuse = torch.cat((p4, p3, p2, p1), 1)
        fuse = self.asf(fuse, [p4, p3, p2, p1])
        fuse = self.d(fuse)
        # 先把fuse的channel降低 之后可以试试在生成fuse时就降低channel

        x = self.up1(c4, c3)
        p1 = torch.cat([x, Upsample(fp, csize[2])], dim=1)
        x = self.up2(self.uni1(p1), c2)
        p2 = torch.cat([x, Upsample(fp, csize[1])], dim=1)
        x = self.up3(self.uni2(p2), torch.add(fuse, c1))
        x = self.up4(x)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        return x