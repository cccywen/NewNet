import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss
import matplotlib.pyplot as plt

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class CriterionDSN(nn.Module):
    # 无辅助分类器
    def __init__(self, class_weight, loss_weight=1.0, ignore_index=255, reduction='mean'):
        super(CriterionDSN, self).__init__()

        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.criterion1 = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index, reduction=reduction)
        if self.aux_classifier:
            self.criterion2 = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):
        # without aux_classifier
        assert preds.shape[2:] == target.shape[1:], f"preds[0] shape must be equality to target."
        return self.criterion1(preds, target)


class BoundaryLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BoundaryLoss, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1).type(
            torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)  # ＜0的就=0

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)


        # for i in range(boudary_targets_pyramid.shape[0]):
        #     # 选择当前图像，去除通道维度
        #     img = boudary_targets_pyramid[i].squeeze()  # 从(1, 512, 512)变为(512, 512)
        #
        #     # 如果你的张量在CUDA上，需要先将其移动到CPU上
        #     img = img.cpu().detach().numpy()
        #     plt.imshow(img, cmap='gray')  # 使用灰度色彩映射
        #     plt.axis('off')
        #     name = '/home/caoyiwen/data/vai_be/be_' + f'Image {i + 1}' + '.png'
        #     plt.savefig(name, format="png")
        #     plt.show()
        #
        # a = torch.sigmoid(boundary_logits)
        # for j in range(a.shape[0]):
        #     # 选择当前图像，去除通道维度
        #     img = a[j].squeeze()  # 从(1, 512, 512)变为(512, 512)
        #
        #     # 如果你的张量在CUDA上，需要先将其移动到CPU上
        #     img = img.cpu().detach().numpy()
        #     plt.imshow(img, cmap='gray')  # 使用灰度色彩映射
        #     plt.axis('off')
        #     name = '/home/caoyiwen/data/vai_be/gn_be_' + f'Image {j + 1}' + '.png'
        #     plt.savefig(name, format="png")
        #     plt.show()

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss+dice_loss


class BESLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss1 = BoundaryLoss()

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            # loss = self.main_loss(logit_main, labels) + 0.2 * self.aux_loss1(logit_aux, labels)
            loss = self.main_loss(logit_main, labels) + 0.1 * self.aux_loss1(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss
