import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Unit(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, shortcut=False):
        super(Conv_Unit, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv3d(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock_3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(BasicBlock_3D, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn_1 = nn.BatchNorm3d(out_channel)
        self.conv3d_2 = nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn_2 = nn.BatchNorm3d(out_channel)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channel)
        )

    def forward(self, x):
        out = F.relu(self.bn_1(self.conv3d_1(x)))
        out = self.bn_2(self.conv3d_2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(BasicBlock_2D, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = F.relu(self.bn_1(self.conv2d_1(x)))
        out = self.bn_2(self.conv2d_2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()

        self.attention_U = nn.Sequential(
            nn.Linear(128, 128),  # matrix U
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # matrix U
            nn.Sigmoid()
        )

    def forward(self, x, w, bias):
        # B*S*d
        A_U = self.attention_U(x)  # B*S*L
        #print('A_U', A_U.shape)
        slice_logits = F.linear(x, w, bias)
        return A_U, slice_logits


class AFFNet(nn.Module):
    def __init__(self):
        super(AFFNet, self).__init__()

        self.conv_1_1 = Conv_Unit(in_channel=1, out_channel=16, kernel_size=3, padding=1, stride=1)
        self.conv_1_2 = Conv_Unit(in_channel=16, out_channel=16, kernel_size=3, padding=1, stride=1)
        self.max_pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.conv_2 = BasicBlock_3D(in_channel=16, out_channel=32, kernel_size=3, padding=1, stride=1)
        self.max_pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.conv_3 = BasicBlock_3D(in_channel=32, out_channel=64, kernel_size=3, padding=1, stride=1)
        self.max_pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # self.conv_4_1 = Conv_Unit(in_channel=64, out_channel=64, kernel_size=1, padding=1, stride=1)
        # self.conv_4_2 = Conv_Unit(in_channel=64, out_channel=64, kernel_size=1, padding=1, stride=1)
        # self.max_pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        # self.conv_5_1 = Conv_Unit(in_channel=64, out_channel=128, kernel_size=1, padding=0, stride=1)
        # self.conv_5_2 = Conv_Unit(in_channel=128, out_channel=128, kernel_size=1, padding=0, stride=1)
        # self.max_pool_5 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        # self.conv_6_1 = Conv_Unit(in_channel=128, out_channel=128, kernel_size=1, padding=0, stride=1)
        # self.conv_6_2 = Conv_Unit(in_channel=128, out_channel=128, kernel_size=1, padding=0, stride=1)

        self.conv_4_v1 = BasicBlock_2D(in_channel=64, out_channel=64, kernel_size=3, padding=1, stride=1)
        self.conv_4_v2 = BasicBlock_2D(in_channel=64, out_channel=64, kernel_size=3, padding=1, stride=1)
        self.conv_4_v3 = BasicBlock_2D(in_channel=64, out_channel=64, kernel_size=3, padding=1, stride=1)

        self.conv_5_v1 = BasicBlock_2D(in_channel=64, out_channel=128, kernel_size=3, padding=1, stride=1)
        self.conv_5_v2 = BasicBlock_2D(in_channel=64, out_channel=128, kernel_size=3, padding=1, stride=1)
        self.conv_5_v3 = BasicBlock_2D(in_channel=64, out_channel=128, kernel_size=3, padding=1, stride=1)

        self.conv_6_v1 = BasicBlock_2D(in_channel=128, out_channel=128, kernel_size=3, padding=1, stride=1)
        self.conv_6_v2 = BasicBlock_2D(in_channel=128, out_channel=128, kernel_size=3, padding=1, stride=1)
        self.conv_6_v3 = BasicBlock_2D(in_channel=128, out_channel=128, kernel_size=3, padding=1, stride=1)

        # self.fc = nn.Linear(in_features=384, out_features=2)
        self.fc = nn.Linear(in_features=128, out_features=2)

        self.fc_view = nn.Linear(in_features=128, out_features=2)

        self.view1_private = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

        self.gated_attention_view1 = GatedAttention()
        self.gated_attention_view2 = GatedAttention()
        self.gated_attention_view3 = GatedAttention()

        self.view2_private = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

        self.view3_private = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

    def view_1(self, out6_pool):
        # B*64*20*24*20-->B*20*64*24*20-->(B*20)*64*24*20
        permute_out6_pool = torch.permute(out6_pool, dims=(0, 2, 1, 3, 4)).contiguous().view(-1, 64, 32, 32)
        #print('permute_out6_pool', permute_out6_pool.shape)
        # 进行2d卷积
        out7 = self.conv_4_v1(permute_out6_pool)
        # (B*20)*128*12*10
        out7_pool = F.max_pool2d(out7, kernel_size=2, stride=2, padding=0)
        # 进行2d卷积
        out8 = self.conv_5_v1(out7_pool)
        # (B*20)*128*6*5
        out8_pool = F.max_pool2d(out8, kernel_size=2, stride=2, padding=0)
        # 进行2d卷积
        out9 = self.conv_6_v1(out8_pool)
        # (B*20)*128*6*5
        #print('out9', out9.shape)
        # (B*20)*128*1*1-->B*20*128
        view1_gap = F.adaptive_avg_pool2d(out9, (1, 1)).view(out6_pool.size(0), -1, 128)
        # print('view1_gap', view1_gap.shape)
        # 进行注意力加权
        # B*20*128-->B*20*1-->B*20*128-->B*128
        attention, slice_logits = self.gated_attention_view1(view1_gap, self.fc_view.weight, self.fc_view.bias)
        v2 = torch.mean(view1_gap * attention, dim=1)
        # v2 = torch.mean(view1_gap, dim=1)
        # print('attention_1', attention_1.shape)
        # 我希望对view1_gap的长度进行约束，使不同view的长度相同。此处先尝试单位化。
        # unit_vector1 = torch.nn.functional.normalize(view1_vector_weighted, p=2, dim=0)
        # 对其进行投影/仿射变换
        v1 = self.view1_private(v2)
        private = v1
        # 计算view1_vector_weighted在private上的投影向量。
        # v1在v2上的投影向量的公式为v1·v2/||v1||^2 * v1
        foo = torch.sum(v2 * v1, keepdim=True, dim=-1) / torch.square(
            torch.norm(v1, keepdim=True, dim=-1))
        projection = foo * v1
        # 使用v2-v1作为common信息
        common = v2 - projection
        return common, private, v2, attention, slice_logits

    def view_2(self, out6_pool):
        # B*64*20*24*20-->B*24*64*20*20-->(B*24)*64*20*20
        permute_out6_pool = torch.permute(out6_pool, dims=(0, 3, 1, 2, 4)).contiguous().view(-1, 64, 32, 32)
        # print('permute_out6_pool', permute_out6_pool.shape)
        out7 = self.conv_4_v2(permute_out6_pool)
        # (B*20)*128*12*10
        out7_pool = F.max_pool2d(out7, kernel_size=2, stride=2, padding=0)
        # 进行2d卷积
        out8 = self.conv_5_v2(out7_pool)
        # (B*20)*128*6*5
        out8_pool = F.max_pool2d(out8, kernel_size=2, stride=2, padding=0)
        # 进行2d卷积
        out9 = self.conv_6_v2(out8_pool)
        # (B*24)*128*1*1-->B*24*128
        view2_gap = F.adaptive_avg_pool2d(out9, (1, 1)).view(out6_pool.size(0), -1, 128)
        # 进行注意力加权
        # B*20*128-->B*20*1-->B*20*128-->B*128
        attention, slice_logits = self.gated_attention_view2(view2_gap, self.fc_view.weight, self.fc_view.bias)
        v2 = torch.mean(view2_gap * attention, dim=1)
        # v2 = torch.mean(view2_gap * attention_2, dim=1)
        # print('attention_1', attention_1.shape)
        # 我希望对view1_gap的长度进行约束，使不同view的长度相同。此处先尝试单位化。
        # unit_vector1 = torch.nn.functional.normalize(view1_vector_weighted, p=2, dim=0)
        # 对其进行投影/仿射变换
        v1 = self.view2_private(v2)
        private = v1
        # 计算view1_vector_weighted在private上的投影向量。
        # v1在v2上的投影向量的公式为v1·v2/||v1||^2 * v1
        foo = torch.sum(v2 * v1, keepdim=True, dim=-1) / torch.square(
            torch.norm(v1, keepdim=True, dim=-1))
        projection = foo * v1
        # 使用v2-v1作为common信息
        common = v2 - projection
        return common, private, v2, attention, slice_logits

    def view_3(self, out6_pool):
        permute_out6_pool = torch.permute(out6_pool, dims=(0, 4, 1, 2, 3)).contiguous().view(-1, 64, 32, 32)
        # print('permute_out6_pool', permute_out6_pool.shape)
        out7 = self.conv_4_v3(permute_out6_pool)
        # (B*20)*128*12*10
        out7_pool = F.max_pool2d(out7, kernel_size=2, stride=2, padding=0)
        # 进行2d卷积
        out8 = self.conv_5_v3(out7_pool)
        # (B*20)*128*6*5
        out8_pool = F.max_pool2d(out8, kernel_size=2, stride=2, padding=0)
        # 进行2d卷积
        out9 = self.conv_6_v3(out8_pool)
        # (B*20)*128*1*1-->B*20*128
        view3_gap = F.adaptive_avg_pool2d(out9, (1, 1)).view(out6_pool.size(0), -1, 128)
        # 进行注意力加权
        # B*20*128-->B*20*1-->B*20*128-->B*128
        attention, slice_logits = self.gated_attention_view3(view3_gap, self.fc_view.weight, self.fc_view.bias)
        v2 = torch.mean(view3_gap * attention, dim=1)
        # print('attention_1', attention_1.shape)
        # 我希望对view1_gap的长度进行约束，使不同view的长度相同。此处先尝试单位化。
        # unit_vector1 = torch.nn.functional.normalize(view1_vector_weighted, p=2, dim=0)
        # 对其进行投影/仿射变换
        v1 = self.view3_private(v2)
        private = v1
        # 计算view1_vector_weighted在private上的投影向量。
        # v1在v2上的投影向量的公式为v1·v2/||v1||^2 * v1
        foo = torch.sum(v2 * v1, keepdim=True, dim=-1) / torch.square(
            torch.norm(v1, keepdim=True, dim=-1))
        projection = foo * v1
        # 使用v2-v1作为common信息
        common = v2 - projection
        return common, private, v2, attention, slice_logits

    def forward(self, x):
        # 160*192*160
        out1 = self.conv_1_1(x)
        out2 = self.conv_1_2(out1)
        out2_pool = self.max_pool_1(out2)
        #¶print('out2_pool', out2_pool.shape)
        # 80*96*80
        out3 = self.conv_2(out2_pool)
        out3_pool = self.max_pool_2(out3)
        #print('out3_pool', out3_pool.shape)
        # 40*48*40
        out4 = self.conv_3(out3_pool)
        out4_pool = self.max_pool_3(out4)
        #print('out4_pool', out4_pool.shape)

        # B*128
        view1common, view1private, view1, attention1, slice_logits1 = self.view_1(out4_pool)  # B*20*1
        view2common, view2private, view2, attention2, slice_logits2 = self.view_2(out4_pool)  # B*24*1
        view3common, view3private, view3, attention3, slice_logits3 = self.view_3(out4_pool)  # B*20*1

        # fusion = torch.concat([view1common, view2common, view3common], dim=-1)
        fusion = (view1common + view2common + view3common) / 3
        # private使用batch内部的进行约束，让其同view一致，不同view远离。
        # B*3*128
        private_fusion = torch.concat([view1private.unsqueeze(dim=1), view2private.unsqueeze(dim=1),
                                       view3private.unsqueeze(dim=1)], dim=1)

        #print('fusion', fusion.shape)
        logits = self.fc(fusion)
        view1 = self.fc_view(view1)
        view2 = self.fc_view(view2)
        view3 = self.fc_view(view3)

        # 基于v2提取一致性信息，对一致性信息进行约束，进而优化v2.
        common_logits1 = self.fc_view(view1common)
        common_logits2 = self.fc_view(view2common)
        common_logits3 = self.fc_view(view3common)

        # print('common_logits1 - view1', common_logits1 - view1)
        # print('common_logits2 - view2', common_logits2 - view2)
        # print('common_logits3 - view3', common_logits3 - view3)

        attention_related = (attention1, attention2, attention3, slice_logits1, slice_logits2, slice_logits3)
        multiview_related = (
            private_fusion, view1, view2, view3, common_logits1 - view1, common_logits2 - view2, common_logits3 - view3)
        return logits, attention_related, multiview_related


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from losses import CrossViewConLoss, LogitLoss
    criterion = nn.CrossEntropyLoss()
    conloss = CrossViewConLoss()
    logit_loss = LogitLoss()
    bag_label = torch.tensor([0.]).long()

    model = AFFNet().eval()
    x = torch.randn(1, 1, 256, 256, 256)
    out1, attention_related, multiview_related = model(x)

    print("logits shape       :", out1.shape)                 # dovrebbe essere [1,1]
    print("attention shapes   :", [t.shape for t in attention_related[:3]])   # es. [1,32], ...
    print("private fusion shape:", multiview_related[0].shape)                 # [1,3,128]

    attention1, attention2, attention3, slice_logits1, slice_logits2, slice_logits3 = attention_related
    private_fusion, view1, view2, view3, diff1, diff2, diff3 = multiview_related

    loss1 = criterion(out1, bag_label)
    view1_loss = criterion(view1, bag_label)
    view2_loss = criterion(view2, bag_label)
    view3_loss = criterion(view3, bag_label)
    # private_fusion:B*3*d
    loss2 = conloss(features=private_fusion)
    # common_loss = simloss(x=common_fusion)

    common1_loss = criterion(diff1, bag_label)
    common2_loss = criterion(diff2, bag_label)
    common3_loss = criterion(diff3, bag_label)
    common_loss = common1_loss + common2_loss + common3_loss

    # 这里的2三分类和四分类要记得修改😍
    attention_loss1 = logit_loss(alpha=attention1, logits=slice_logits1.view(-1, 2),
                                    target=bag_label.repeat(slice_logits1.size(1), 1).permute(1, 0).contiguous().view(
                                        -1))
    attention_loss2 = logit_loss(alpha=attention2, logits=slice_logits2.view(-1, 2),
                                    target=bag_label.repeat(slice_logits2.size(1), 1).permute(1, 0).contiguous().view(
                                        -1))
    attention_loss3 = logit_loss(alpha=attention3, logits=slice_logits3.view(-1, 2),
                                    target=bag_label.repeat(slice_logits3.size(1), 1).permute(1, 0).contiguous().view(
                                        -1))

    attention_loss = attention_loss1 + attention_loss2 + attention_loss3
    print('loss1', loss1)
    print('loss2', loss2)
    print('view1_loss', view1_loss)
    print('view2_loss', view2_loss)
    print('view3_loss', view3_loss)
    print('common_loss', common_loss)
    # print('commonxx_loss', commonxx_loss)
    # loss = loss1 + loss2 + loss3
    loss = loss1 + 0.4*loss2 + (view1_loss + view2_loss + view3_loss) + 0.4*common_loss + attention_loss
    print('out1' + str(out1) + 'bag_label' + str(bag_label))