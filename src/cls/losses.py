from __future__ import print_function

import torch
import torch.nn as nn
import math

class CrossViewConLoss(nn.Module):

    def __init__(self, ):
        super(CrossViewConLoss, self).__init__()
        self.temperature = 1

    def forward(self, features):
        # feature的shape为B*3*d
        batch, view, dimension = features.size()
        # 1.首先对其进行转置和view操作 B*3*d-->3*B*d-->(3B)*d
        features = torch.permute(features, dims=(1, 0, 2)).contiguous().view(-1, dimension)
        # 2.计算余弦相似度矩阵
        x = features.unsqueeze(1)
        y = features.unsqueeze(0)
        cosine_similarities = torch.cosine_similarity(x, y, dim=-1)
        # 3.求绝对值
        abs_cosine = torch.abs(cosine_similarities)
        print('cosine_similarities', cosine_similarities)
        # 4.对每个位置求指数
        exp_cosine_similarities = torch.exp(cosine_similarities)
        exp_abs_cosine = torch.exp(abs_cosine)
        print('exp_cosine_similarities max', exp_cosine_similarities.max())
        print('exp_cosine_similarities min', exp_cosine_similarities.min())
        # 5.求分母(基于绝对值求分母)
        denominator = torch.sum(exp_abs_cosine, dim=-1)
        # 6.生成一个mask矩阵
        ones = torch.ones((batch, batch)).cuda()
        mask = torch.block_diag(ones, ones, ones)
        # 7.mask与指数后的相似度矩阵相乘，基于原始相似度求分子。
        masked_exp = mask * exp_cosine_similarities
        # 8.求分子
        numerator = torch.sum(masked_exp, dim=-1)
        # 9.分子除以分母然后取log，求和，除以batch，添加负号
        loss = -(torch.sum(torch.log(numerator / denominator)) / batch)

        return loss


def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    # 450 2
    # 450表示patch数 2表示类别
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        # torch.arange(0, 2).long().repeat(450, 1) 输出是450*2
        # target.data.repeat(num_classes, 1) 输出是2*450
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    # 450*1
    # 根据one-hot将所属类别对应的logit提取出来.
    return logits.masked_select(one_hot_mask)


class LogitLoss(nn.Module):
    def __init__(self):
        super(LogitLoss, self).__init__()

    # alpha:3*150
    # logits:450*2
    # target:450
    # batch_size:3
    # ppt步骤3
    def forward(self, alpha, logits, target):
        # 模仿CrossEntropyLoss，先基于logit和target获取真实类别所对应的logit。
        # selected_logits:450*1->3*150
        # ppt步骤1 送入到class_select函数之前先进行softmax。
        alpha = alpha.view(alpha.size(0), alpha.size(1))
        print('alpha', alpha)
        softmax_logits = torch.softmax(logits, dim=1)
        print('softmax_logits', softmax_logits.shape)
        selected_logits = class_select(softmax_logits, target).view(alpha.shape[0], -1)
        print('selected_logits', selected_logits.shape)
        # 3*1
        # logits_sum = torch.sum(selected_logits, dim=1, keepdim=True)
        # print('logits_sum',logits_sum.shape)
        # 3*150
        # norm_logits = torch.div(selected_logits, logits_sum)
        # print('norm_logits',norm_logits.shape)
        # 3*150->3->1
        # logit_loss = torch.norm(input=alpha - norm_logits, p=2, dim=-1).sum()
        # logit_loss = torch.norm(input=torch.relu(torch.abs(alpha - selected_logits) - 0.2), p=2, dim=-1).sum()
        # print('logit_loss', logit_loss)

        logit_loss = torch.nn.functional.kl_div(torch.log_softmax(alpha, dim=-1),
                                                torch.softmax(selected_logits, dim=-1), reduction='batchmean')
        print('logit_loss', logit_loss)
        return logit_loss


class AFFNetLoss(nn.Module):
    def __init__(self, weight2=0.4, weight_common=0.4):
        super(AFFNetLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.conloss = CrossViewConLoss()
        self.logit_loss = LogitLoss()
        self.weight2 = weight2
        self.weight_common = weight_common

    def forward(self, model_outputs, target):
        # model_outputs is expected to be (logits, attention_related, multiview_related)
        # as returned by AFFNet.forward()
        logits, attention_related, multiview_related = model_outputs
        
        # bag_label should be long for CrossEntropy
        bag_label = target.long().view(-1)
        
        # 1. Main classification loss
        loss1 = self.ce(logits, bag_label)
        
        # 2. View specific classification losses
        attention1, attention2, attention3, slice_logits1, slice_logits2, slice_logits3 = attention_related
        private_fusion, view1, view2, view3, diff1, diff2, diff3 = multiview_related
        
        view1_loss = self.ce(view1, bag_label)
        view2_loss = self.ce(view2, bag_label)
        view3_loss = self.ce(view3, bag_label)
        
        # 3. Consistency loss (private fusion)
        loss2 = self.conloss(features=private_fusion)
        
        # 4. Common logits loss
        common1_loss = self.ce(diff1, bag_label)
        common2_loss = self.ce(diff2, bag_label)
        common3_loss = self.ce(diff3, bag_label)
        common_loss = common1_loss + common2_loss + common3_loss
        
        # 5. Attention/Saliency loss
        # slice_logits shape: (B, S, 1) usually, but LogitLoss expected (B, 2)
        # In AFFNet.py, it uses slice_logits.view(-1, 2)
        attention_loss1 = self.logit_loss(alpha=attention1, logits=slice_logits1.view(-1, 2),
                                         target=bag_label.repeat(slice_logits1.size(1), 1).permute(1, 0).contiguous().view(-1))
        attention_loss2 = self.logit_loss(alpha=attention2, logits=slice_logits2.view(-1, 2),
                                         target=bag_label.repeat(slice_logits2.size(1), 1).permute(1, 0).contiguous().view(-1))
        attention_loss3 = self.logit_loss(alpha=attention3, logits=slice_logits3.view(-1, 2),
                                         target=bag_label.repeat(slice_logits3.size(1), 1).permute(1, 0).contiguous().view(-1))
        
        attention_loss = attention_loss1 + attention_loss2 + attention_loss3
        
        # Total loss
        total_loss = loss1 + self.weight2 * loss2 + (view1_loss + view2_loss + view3_loss) + \
                     self.weight_common * common_loss + attention_loss
        
        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs can be (B, 1) or (B) -> raw logits
        # targets can be (B, 1) or (B) -> 0 or 1
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        # preds, targets: (B, 4) -> [x, y, w, h] normalized
        xA = torch.max(preds[:, 0], targets[:, 0])
        yA = torch.max(preds[:, 1], targets[:, 1])
        xB = torch.min(preds[:, 0] + preds[:, 2], targets[:, 0] + targets[:, 2])
        yB = torch.min(preds[:, 1] + preds[:, 3], targets[:, 1] + targets[:, 3])

        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
        areaPred = preds[:, 2] * preds[:, 3]
        areaTarget = targets[:, 2] * targets[:, 3]
        unionArea = areaPred + areaTarget - interArea + 1e-8
        iou = interArea / unionArea

        cp_x = preds[:, 0] + preds[:, 2] / 2
        cp_y = preds[:, 1] + preds[:, 3] / 2
        ct_x = targets[:, 0] + targets[:, 2] / 2
        ct_y = targets[:, 1] + targets[:, 3] / 2
        rho2 = (cp_x - ct_x)**2 + (cp_y - ct_y)**2

        cxA = torch.min(preds[:, 0], targets[:, 0])
        cyA = torch.min(preds[:, 1], targets[:, 1])
        cxB = torch.max(preds[:, 0] + preds[:, 2], targets[:, 0] + targets[:, 2])
        cyB = torch.max(preds[:, 1] + preds[:, 3], targets[:, 1] + targets[:, 3])
        c2 = (cxB - cxA)**2 + (cyB - cyA)**2 + 1e-8

        v = (4 / (math.pi**2)) * torch.pow(
            torch.atan(targets[:, 2] / (targets[:, 3] + 1e-8)) - 
            torch.atan(preds[:, 2] / (preds[:, 3] + 1e-8)), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-8)
        
        ciou = iou - (rho2 / c2) - alpha * v
        loss = 1 - ciou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss