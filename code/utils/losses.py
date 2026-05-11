import torch
from torch.nn import functional as F
import torch.nn as nn
import contextlib
import pdb
import numpy as np
import math

class MaskAMOSDiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5, include_background=False, from_logits=True):
        super().__init__()
        self.nclass = nclass
        self.smooth = float(smooth)
        self.include_background = include_background
        self.from_logits = from_logits

        if class_weights is None:
            cw = torch.ones(1, nclass, dtype=torch.float32)
        else:
            cw = torch.as_tensor(class_weights, dtype=torch.float32).view(1, nclass)
            assert cw.numel() == nclass, "class_weights length must equal nclass"
        # 不训练的 buffer/parameter 都可；这里用 register_buffer 更合适
        self.register_buffer("class_weights", cw)

    def forward(self, preds, target, mask=None):
        """
        preds: [B, C, ...] (logits if from_logits=True, else probabilities)
        target: [B, 1, ...] 或 [B, ...] (整型类别索引)
        mask: [B, 1, ...] 或 [B, ...] (1/0 有效区域；可选)
        """
        B, C = preds.shape[:2]
        # 统一形状：展平到 [B, C, N]
        if self.from_logits:
            probs = torch.softmax(preds, dim=1)
        else:
            probs = preds
        probs = probs.view(B, C, -1)

        if target.dim() == preds.dim():
            target = target.squeeze(1)   # 兼容 [B,1,...]
        target = target.view(B, -1).long()

        # one-hot: [B, C, N]
        target_oh = torch.zeros_like(probs)
        target_oh.scatter_(1, target.unsqueeze(1), 1.0)

        if not self.include_background and C > 1:
            probs = probs[:, 1:, :]
            target_oh = target_oh[:, 1:, :]
            cw = self.class_weights[:, 1:]
        else:
            cw = self.class_weights

        if mask is not None:
            if mask.dim() == preds.dim():
                mask = mask.squeeze(1)
            mask = mask.view(B, -1).float()
            # 扩到类维度 [B, C', N]
            mask = mask.unsqueeze(1)
            probs_m = probs * mask
            tgt_m = target_oh * mask
        else:
            probs_m = probs
            tgt_m = target_oh

        # per-class intersection & denominator: [B, C']
        inter = (probs_m * tgt_m).sum(dim=2)
        denom = probs_m.sum(dim=2) + tgt_m.sum(dim=2)

        dice_c = (2 * inter + self.smooth) / (denom + self.smooth)  # [B, C']

        # 排除 batch 中完全不存在的类别（gt 全零 → denom ≈ smooth）
        # 当类别不存在时，Dice 会被 smooth 人为抬高到 1.0，导致 loss 反而奖励预测全背景
        present = (tgt_m.sum(dim=2) > self.smooth).float()  # [B, C'] 1=该类存在
        dice_c = dice_c * present  # 不存在的类 Dice 置 0

        # 加权平均（仅对存在的类别求平均）
        cw_safe = cw.clamp_min(1e-12)
        present_cw = cw_safe * present  # [B, C']
        cw_sum = present_cw.sum(dim=1).clamp_min(1e-12)  # [B]
        dice_w = (dice_c * cw_safe).sum(dim=1) / cw_sum  # [B]
        loss = 1.0 - dice_w.mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss: -alpha_t * (1-p_t)^gamma * log(p_t)

    Handles class imbalance by focusing on hard examples.
    Unlike CE, provides gradient even when model predicts zero for a class.

    Args:
        gamma: focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha: per-class weight tensor of shape (num_classes,). If None, uniform weights.
        reduction: 'mean' (default), 'sum', or 'none'
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        """
        logits: [B, C, ...] raw logits
        target: [B, ...] long tensor of class indices
        Returns: scalar loss (mean) if reduction='mean', else [B, ...] per-element
        """
        B, C = logits.shape[:2]
        spatial = logits.shape[2:]
        logits_flat = logits.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)  # [B*N, C]
        target_flat = target.view(-1).long()  # [B*N]

        log_probs = F.log_softmax(logits_flat, dim=1)  # [B*N, C]
        probs = log_probs.exp()  # [B*N, C]

        # gather the probability of the true class
        pt = probs.gather(1, target_flat.unsqueeze(1)).squeeze(1)  # [B*N]
        log_pt = log_probs.gather(1, target_flat.unsqueeze(1)).squeeze(1)  # [B*N]

        # focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma  # [B*N]

        # per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target_flat)  # [B*N]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt  # [B*N]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # reshape back to [B, *spatial] to match nn.CrossEntropyLoss(reduction='none')
            return loss.view(B, *spatial)


class MultiLabelFocalLoss(nn.Module):
    """Per-channel binary Focal Loss for 15 independent organ heads.

    Each channel (1..15) is a binary classifier with sigmoid activation.
    No softmax — each organ competes only against itself.
    Background is channel 0 (also sigmoid, but loss weight can be reduced).
    """
    def __init__(self, gamma=2.0, alpha=None, bg_weight=0.1, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.bg_weight = bg_weight
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32).view(1, -1))
        else:
            self.alpha = None

    def forward(self, logits, target):
        """
        logits: [B, C, ...] raw logits (C=16 includes background)
        target: [B, ...] long class indices
        """
        B, C = logits.shape[:2]
        spatial = logits.shape[2:]
        logits_flat = logits.view(B, C, -1)  # [B, C, N]

        # Convert class target to multi-label: [B, C, N]
        target_ml = torch.zeros_like(logits_flat)
        target_flat = target.view(B, -1).unsqueeze(1)  # [B, 1, N]
        target_ml.scatter_(1, target_flat, 1.0)

        # Binary Focal Loss per channel
        probs = torch.sigmoid(logits_flat)  # [B, C, N]
        # p_t = p if target=1 else 1-p
        pt = target_ml * probs + (1 - target_ml) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits_flat, target_ml, reduction='none')

        loss = focal_weight * bce  # [B, C, N]

        # Per-channel alpha and background down-weighting
        w = torch.ones(1, C, 1, device=logits.device)
        w[:, 0, :] = self.bg_weight  # down-weight background channel
        if self.alpha is not None:
            w = w * self.alpha.view(1, -1, 1)
        loss = loss * w

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # Return per-voxel sum over channels: [B, *spatial]
            return loss.sum(dim=1).view(B, *spatial)


class MultiLabelDiceLoss(nn.Module):
    """Per-channel binary Dice Loss for 15 independent organ heads."""
    def __init__(self, nclass, class_weights=None, bg_weight=0.1, smooth=1e-5):
        super().__init__()
        self.nclass = nclass
        self.smooth = smooth
        self.bg_weight = bg_weight
        if class_weights is None:
            cw = torch.ones(1, nclass)
        else:
            cw = torch.as_tensor(class_weights, dtype=torch.float32).view(1, nclass)
        cw[0, 0] = bg_weight  # down-weight background
        self.register_buffer("class_weights", cw)

    def forward(self, logits, target, mask=None):
        """
        logits: [B, C, ...] raw logits
        target: [B, ...] long class indices
        mask: optional region mask [B, ...]
        """
        B, C = logits.shape[:2]
        logits_flat = logits.view(B, C, -1)
        probs = torch.sigmoid(logits_flat)  # [B, C, N]

        # Multi-label target
        target_ml = torch.zeros_like(probs)
        target_flat = target.view(B, -1).unsqueeze(1)
        target_ml.scatter_(1, target_flat, 1.0)

        if mask is not None:
            if mask.dim() == logits.dim():
                mask = mask.squeeze(1)
            mask = mask.view(B, 1, -1).float()
            probs = probs * mask
            target_ml = target_ml * mask

        # Per-channel binary Dice (skip background C=0)
        inter = (probs[:, 1:, :] * target_ml[:, 1:, :]).sum(dim=2)  # [B, C-1]
        denom = probs[:, 1:, :].sum(dim=2) + target_ml[:, 1:, :].sum(dim=2)  # [B, C-1]
        dice_c = (2 * inter + self.smooth) / (denom + self.smooth)
        present = (target_ml[:, 1:, :].sum(dim=2) > self.smooth).float()
        dice_c = dice_c * present

        cw = self.class_weights[:, 1:]
        cw_present = cw * present
        dice_w = (dice_c * cw).sum(dim=1) / cw_present.sum(dim=1).clamp_min(1e-12)
        return 1.0 - dice_w.mean()


class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceLossMulti(nn.Module):
    """
    Multi-class Dice Loss with optional class weights, mask support and
    absent-class handling.

    Args
    ----
    nclass : int
        Total number of categories (including background).
    class_weights : sequence or Tensor or None
        Per-class weight. If None, all ones.
    ignore_index : int or None
        Class ID to ignore in Dice averaging (e.g., background=0).
    smooth : float
        Laplace smoothing constant to avoid NaN.
    """
    def __init__(self, nclass,
                 class_weights=None,
                 ignore_index=None,
                 smooth=1e-5):
        super().__init__()
        self.nclass = nclass
        self.ignore_index = ignore_index
        self.smooth = smooth

        if class_weights is None:
            w = torch.ones(nclass, dtype=torch.float32)
        else:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            assert w.shape[0] == nclass, "class_weights length must == nclass"
        self.register_buffer("class_weights", w)

    # ----------------------------------------------------------
    # helper - flatten (N,C,*) tensor to (N,C,L)
    # ----------------------------------------------------------
    @staticmethod
    def _flatten(t):
        N, C = t.shape[:2]
        return t.view(N, C, -1)

    def forward(self, logits, target, mask=None):
        """
        Parameters
        ----------
        logits : Tensor, shape (N, C, ...)
        target : LongTensor, shape (N, H, W, ...)
        mask   : Bool / Float Tensor, shape (N, 1, ...) or None
                 1 = valid pixel, 0 = ignore pixel

        Returns
        -------
        Dice loss (scalar)
        """
        N, C = logits.shape[:2]
        assert C == self.nclass, "logits C dim mismatch nclass"

        # ---- 1. probability + one-hot ----
        probs = F.softmax(self._flatten(logits), dim=1)      # (N,C,L)
        tgt_oh = F.one_hot(target.view(N, -1), num_classes=C)  # (N,L,C)
        tgt_oh = tgt_oh.permute(0, 2, 1).float()             # (N,C,L)

        # ---- 2. intersection / union ----
        inter = probs * tgt_oh
        union = probs + tgt_oh

        if mask is not None:
            mask = mask.view(N, 1, -1).float()
            inter = inter * mask
            union = union * mask

        # (N,C)
        inter = inter.sum(-1)
        union = union.sum(-1)

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)  # (N,C)

        # ---- 3. absent-class handling ----
        present = (tgt_oh.sum(-1) > 0).float()             # (N,C) 1=class appears
        dice = dice * present                              # absent→0

        # ---- 4. weighting & averaging ----
        w = self.class_weights.unsqueeze(0)                # (1,C)
        if self.ignore_index is not None:
            w = w.clone()
            w[..., self.ignore_index] = 0                  # ignore background weight

        dice = (dice * w).sum() / (present * w).sum().clamp_min(1.0)

        return 1.0 - dice

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            # bug found by @CamillerFerros at github issue#25
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes): 
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


class CrossEntropyLoss(nn.Module):
    def __init__(self, n_classes):
        super(CrossEntropyLoss, self).__init__()
        self.class_num = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.class_num):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.class_num):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _ce_loss(slef, score, target, mask):
        target = target.float()
        loss = (-target * torch.log(score) * mask.float()).sum() / (mask.sum() + 1e-16)
        return loss

    def forward(self, inputs, target, mask, weight=None):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        mask = self._one_hot_mask_encoder(mask)
        loss = 0.0
        for i in range(0, self.class_num):
            if weight is not None:
                loss += self._ce_loss(inputs[:,i], target[:, i], mask[:, i]) * weight
            # loss += self._ce_loss(inputs[:,i], target[:, i], mask[:, i])
        return loss / self.class_num 

def bce_weight(soft_label):
    prob = torch.max(soft_label,dim=1)[0]
    #initialize Gaussian mean and variance
    u1 = 0.5
    sigma1 = 0.1
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma1))
    right = np.exp(-(prob.detach().cpu().numpy() - u1)**2 / (2 * sigma1))
    weight_numpy = 1.3 - left*right
    weight = torch.from_numpy(weight_numpy).cuda()
    return weight

def bce_weight2(soft_label,num_classes):
    x1 = torch.topk(soft_label, 2, dim=1)[0][:,0,...]
    x2 = torch.topk(soft_label, 2, dim=1)[0][:,1,...]
    tau = x1 - ((1 - x1) / (num_classes - 1))
    rt = x1 - x2
    w = torch.where(rt >= tau, torch.ones_like(rt), rt / tau)


    return w



class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.CEloss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, inputs, targets,weight=None):
        loss = self.CEloss(inputs, targets)
        if weight is not None:
            loss = loss * weight
        loss = torch.mean(loss)
        return loss

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass

class Dice_Loss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes
        
@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d

class VAT2d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)

    def forward(self, model, x):
        with torch.no_grad():
            pred= F.softmax(model(x)[0], dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d) 
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                logp_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)[0]
            logp_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(logp_hat, pred)
        return lds

class VAT3d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss
        
    def forward(self, model, x):
        with torch.no_grad():
            pred= F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device) ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d) ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, pred)
        return lds

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

def geometric_loss(out_unl, plab_a, lab_a, loss_mask):
    mask_a = out_unl
    mask_b = plab_a * loss_mask + lab_a * (1 - loss_mask)
    # 两输入都是(b,256,256)
    # # 可视化
    # import matplotlib.pyplot as plt
    # mask_a_ = mask_a[0].detach().cpu().numpy()  # 将第一个图像从GPU移到CPU，并转换为numpy数组
    # mask_b_ = mask_b[0].detach().cpu().numpy()  # 同理，处理第二个掩码图像
    # # 创建2x2子图布局
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # # 可视化第一个掩码
    # axes[0].imshow(mask_a_, cmap='gray')
    # axes[0].set_title('out Mask A')
    # axes[0].axis('off')  # 去掉坐标轴
    # # 可视化第二个掩码
    # axes[1].imshow(mask_b_, cmap='gray')
    # axes[1].set_title('mix Mask B')
    # axes[1].axis('off')  # 去掉坐标轴
    # # 显示图像
    # plt.show()

    mask_a = mask_a.float()
    mask_b = mask_b.float()
    mask_a = mask_a.unsqueeze(1)  # 在第二维插入一个维度，变成 (b, 1, 256, 256)
    mask_b = mask_b.unsqueeze(1)

    # 定义Sobel算子
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).float().unsqueeze(0).unsqueeze(0).cuda()  # (1,1,3,3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float().unsqueeze(0).unsqueeze(0).cuda()

    # 计算梯度
    grad_a_x = F.conv2d(mask_a, sobel_x, stride=1, padding=1)
    grad_a_y = F.conv2d(mask_a, sobel_y, stride=1, padding=1)
    grad_a = torch.sqrt(grad_a_x ** 2 + grad_a_y ** 2)  # 梯度大小，表示边缘强度

    grad_b_x = F.conv2d(mask_b, sobel_x, stride=1, padding=1)
    grad_b_y = F.conv2d(mask_b, sobel_y, stride=1, padding=1)
    grad_b = torch.sqrt(grad_b_x ** 2 + grad_b_y ** 2)  # 梯度大小，表示边缘强度
    # 可视化
    # # 将输出张量从 GPU 转移到 CPU，并转换为 NumPy 数组
    # grad_a_x_cpu = grad_a_x.squeeze(0).cpu().detach().numpy()  # 去掉批次维度，并转到 CPU
    # grad_a_y_cpu = grad_a_y.squeeze(0).cpu().detach().numpy()
    # grad_a_cpu = grad_a.squeeze(0).cpu().detach().numpy()
    #
    # grad_b_x_cpu = grad_b_x.squeeze(0).cpu().detach().numpy()
    # grad_b_y_cpu = grad_b_y.squeeze(0).cpu().detach().numpy()
    # grad_b_cpu = grad_b.squeeze(0).cpu().detach().numpy()
    #
    # # 可视化 grad_a_x, grad_a_y, grad_a, grad_b_x, grad_b_y, grad_b
    # plt.figure(figsize=(18, 12))
    #
    # # 第一行：显示 grad_a 系列
    # # 可视化 grad_a_x
    # plt.subplot(2, 3, 1)
    # plt.imshow(grad_a_x_cpu[0].squeeze(0), cmap='gray')  # 显示第一个通道（灰度图）
    # plt.title("grad_a_x (X direction)")
    # plt.axis('off')
    #
    # # 可视化 grad_a_y
    # plt.subplot(2, 3, 2)
    # plt.imshow(grad_a_y_cpu[0].squeeze(0), cmap='gray')  # 显示第一个通道（灰度图）
    # plt.title("grad_a_y (Y direction)")
    # plt.axis('off')
    #
    # # 可视化 grad_a
    # plt.subplot(2, 3, 3)
    # plt.imshow(grad_a_cpu[0].squeeze(0), cmap='gray')  # 显示第一个通道（灰度图）
    # plt.title("grad_a (Magnitude)")
    # plt.axis('off')
    #
    # # 第二行：显示 grad_b 系列
    # # 可视化 grad_b_x
    # plt.subplot(2, 3, 4)
    # plt.imshow(grad_b_x_cpu[0].squeeze(0), cmap='gray')  # 显示第一个通道（灰度图）
    # plt.title("grad_b_x (X direction)")
    # plt.axis('off')
    #
    # # 可视化 grad_b_y
    # plt.subplot(2, 3, 5)
    # plt.imshow(grad_b_y_cpu[0].squeeze(0), cmap='gray')  # 显示第一个通道（灰度图）
    # plt.title("grad_b_y (Y direction)")
    # plt.axis('off')
    #
    # # 可视化 grad_b
    # plt.subplot(2, 3, 6)
    # plt.imshow(grad_b_cpu[0].squeeze(0), cmap='gray')  # 显示第一个通道（灰度图）
    # plt.title("grad_b (Magnitude)")
    # plt.axis('off')
    #
    # plt.show()
    # 计算几何一致性损失：这里使用MSE损失来比较梯度差异
    geo_loss = F.mse_loss(grad_a, grad_b)  # 计算梯度差异的均方误差

    return geo_loss
