from nets.unet import UNet, UNet_2d
from nets.VNet import VNet
import torch.nn as nn
import torch
import torch.nn.functional as F

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", ema=False):
    if net_type == "unet" and mode == "train":
        net = UNetWithPCA(net_type, in_chns=in_chns,class_num=class_num, num_components=3, threshold=0.9).cuda()
    if net_type == "VNet" and mode == "train":
        net = UNetWithPCA(net_type,in_chns=in_chns, class_num=class_num, num_components=3, threshold=0.6).cuda()
    if net_type == "VNet" and mode == "test":
        net = UNetWithPCA(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def BCP_net(net_type, in_chns=1, class_num=2, ema=False):
    if net_type == "UNETR":
        net = UNetWithPCA(net_type, in_chns=in_chns, class_num=class_num, num_components=3, threshold=0.6).cuda()
    else:
        net = UNetWithPCA(net_type, in_chns=in_chns,class_num=class_num, num_components=3, threshold=0.9).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


class UNetWithPCA(nn.Module):
    def __init__(self,net_type, in_chns=1, class_num=4, num_components=3, threshold=0.5):
        super(UNetWithPCA, self).__init__()
        self.num_classes = class_num
        self.num_components = num_components
        self.threshold = threshold
        self.net_type = net_type

        # 定义UNet模型的结构
        if self.net_type == "unet":
            self.unet = UNet_2d(in_chns=in_chns, class_num=self.num_classes).cuda()
        elif self.net_type == "UNETR":
            self.unetr = UNETR(
                in_channels=in_chns,
                out_channels=self.num_classes,
                img_size=(96, 96, 64),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0,
            ).cuda()
        else:
            self.vnet = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
        self.dim = 16


        # 初始化主成分矩阵W，每个类别有3个主成分，假设特征维度为C
        self.W = nn.Parameter(torch.randn(self.num_classes, self.dim, num_components))  # 每个类别有3个主成分
        # 最后一层的线性分类头
        # self.classification_head = nn.Conv2d(16, self.num_classes, kernel_size=1)

    def forward(self, x, pcaW=None):
        if self.net_type == "unet":
            logits, feature, features = self.unet(x, pcaW)
        elif self.net_type == "UNETR":
            logits, features = self.unetr(x)
        else:
            logits, features = self.vnet(x, pcaW)

        prob_map = F.softmax(logits, dim=1)  # 预测的类别概率
        confidence_map, predicted_class = torch.max(prob_map, dim=1)  # 最终预测的类别 (batch_size, height, width)
        reconfidence_map, tau = self.relativeConfindence(prob_map)

        A_features, B_features, A_class_positions, B_class_positions = self.partition_features(features, predicted_class, reconfidence_map, tau)
        A_features2, A_class_positions2 = self.partition_features2(features, predicted_class, reconfidence_map, tau)
        A_features_combined = self.add_features(A_features, A_features2)
        updated_W, loss = self.update_principal_components(A_features_combined)

        return logits, updated_W, loss#, features


    def relativeConfindence(self, prob_map):
        x1 = torch.topk(prob_map, 2, dim=1)[0][:, 0, ...]
        x2 = torch.topk(prob_map, 2, dim=1)[0][:, 1, ...]
        tau = x1-((1-x1)/(self.num_classes-1))-0.2
        rt= x1 -x2
        return rt, tau

    def add_features(self, A_features1, A_features2):
        """将两个特征字典按类别相加"""
        result = {}
        for class_idx in A_features1.keys():
            if class_idx in A_features2 and len(A_features1[class_idx]) > 0 and len(A_features2[class_idx]) > 0:
                # 确保特征形状相同
                if A_features1[class_idx].shape == A_features2[class_idx].shape:
                    result[class_idx] = (A_features1[class_idx] + A_features2[class_idx]) / 2
                else:
                    # 如果形状不同，选择其中一个（或者根据需求处理）
                    result[class_idx] = A_features1[class_idx]  # 或者 A_features2[class_idx]
            else:
                # 如果只有一个字典有该类别特征，直接使用
                result[class_idx] = A_features1[class_idx] if len(A_features1[class_idx]) > 0 else A_features2.get(
                    class_idx, torch.tensor([]))
        return result


    def partition_features(self, features, predicted_class, confidence_map, tau):
        if len(features.shape) == 4:
            B, D, H, W = features.shape
        else:
            B, D, H, W, Z = features.shape

        # 初始化 A 类和 B 类特征存储
        A_class_features = {i: [] for i in range(self.num_classes)}  # 每个类别的 A 类特征
        B_class_features = {i: [] for i in range(self.num_classes)}  # 每个类别的 B 类特征
        A_class_positions = {i: [] for i in range(self.num_classes)}
        B_class_positions = {i: [] for i in range(self.num_classes)}
        # 展平特征和对应的预测类别及置信度
        if len(features.shape) == 4:
            features = features.permute(0, 2, 3, 1).reshape(-1, D)  # 转换为 (B * H * W, D)
        else:
            features = features.permute(0, 2, 3, 4, 1).reshape(-1, D)  # 转换为 (B * H * W * Z, D)
        predicted_class = predicted_class.flatten()  # 转换为 (B * H * W,)
        confidence_map = confidence_map.flatten()  # 转换为 (B * H * W,)
        tau = tau.flatten()  # 转换为 (B * H * W,)

        # 遍历每个类别进行分类
        for class_idx in range(self.num_classes):
            # 当前类别的掩码
            class_mask = (predicted_class == class_idx)

            # A 类特征：置信度高于阈值的特征
            # A_mask = class_mask & (confidence_map > self.threshold)
            A_mask = class_mask & (confidence_map > tau)
            A_positions = torch.nonzero(A_mask)
            A_features = features[A_mask]  # 提取该类别的 A 类特征
            if len(A_features) > 0:
                A_class_features[class_idx] = A_features
                A_class_positions[class_idx] = A_positions

            # B 类特征：置信度低于等于阈值的特征
            B_mask = class_mask & (confidence_map <= tau)
            B_positions = torch.nonzero(B_mask)
            B_features = features[B_mask]  # 提取该类别的 B 类特征
            if len(B_features) > 0:
                B_class_features[class_idx] = B_features
                B_class_positions[class_idx] = B_positions

        return A_class_features, B_class_features, A_class_positions, B_class_positions
    def partition_features2(self, features, predicted_class, confidence_map, tau):
        if len(features.shape) == 4:
            B, D, H, W = features.shape
        else:
            B, D, H, W, Z = features.shape

        # 初始化 A 类特征存储
        A_class_features = {i: [] for i in range(self.num_classes)}  # 每个类别的 A 类特征
        A_class_positions = {i: [] for i in range(self.num_classes)}
        # 展平特征和对应的预测类别及置信度
        if len(features.shape) == 4:
            features = features.permute(0, 2, 3, 1).reshape(-1, D)  # 转换为 (B * H * W, D)
        else:
            features = features.permute(0, 2, 3, 4, 1).reshape(-1, D)  # 转换为 (B * H * W * Z, D)
        predicted_class = predicted_class.flatten()  # 转换为 (B * H * W,)
        confidence_map = confidence_map.flatten()  # 转换为 (B * H * W,)

        # 遍历每个类别进行分类
        for class_idx in range(self.num_classes):
            # 当前类别的掩码
            class_mask = (predicted_class == class_idx)

            # A 类特征：所有特征，但加权（特征乘以置信度）
            A_mask = class_mask  # 使用整个类别的掩码，不进行置信度筛选
            A_positions = torch.nonzero(A_mask)
            A_features = features[A_mask]  # 提取该类别的所有特征

            if len(A_features) > 0:
                # 获取对应的置信度并扩展维度以便广播
                A_confidence = confidence_map[A_mask].unsqueeze(1)  # 形状: [n_features, 1]
                # 特征加权：特征乘以对应的置信度
                A_features_weighted = A_features * A_confidence
                A_class_features[class_idx] = A_features_weighted
                A_class_positions[class_idx] = A_positions
        return A_class_features, A_class_positions

    def update_principal_components(self, A_features):
        """
        更新 A 类特征的主成分。
        假设 A_features 是 (batch_size, height, width) 的张量，包含 A 类特征的索引。
        """
        # A_features 是一个包含特征类别索引的张量，我们需要对其按类别进行聚合
        # 将 A 类特征提取出来，假设 A_features 中的每个元素是类别索引

        # 获取每个类别的特征
        principal_components = []
        loss_total = 0
        for class_idx in range(self.num_classes):
            class_features = A_features[class_idx] # 获取当前类别的 A 类特征位置
            # 根据 A 类特征学习该类别的主成分（假设通过拉格朗日乘子法进行优化）
            if len(class_features) > 0:
                loss, update_W = self.optimize_principal_component(class_features, class_idx)
                loss_total += loss
                principal_components.append(update_W)
            else:
                principal_components.append(torch.ones(self.dim, self.num_components).cuda())

        # 将每个类别的主成分矩阵拼接为一个张量
        updated_W = torch.stack(principal_components, dim=0)

        return updated_W, loss_total

    def optimize_principal_component(self, features, class_idx):
        """
        使用拉格朗日乘子法更新类别的主成分。
        假设 features 是该类别 A 类特征的张量（形状为 [num_features, feature_dim]）。
        """

        W = self.W[class_idx]
        projected = torch.matmul(features, W)
        reprojected = torch.matmul(projected, W.T)
        total_loss1 = torch.log(torch.mean((reprojected - features) ** 2)+1)  # 投影误差

        return total_loss1, W

    def update_features_with_principal_components(self, A_class_features, W):
        """
        使用更新后的主成分来更新 A 类特征。

        参数:
        - A_class_features: dict，每个类别的 A 类特征 {类别索引: 特征张量 (N_class, D)}
        - W: torch.Tensor，形状为 (num_classes, D, num_components)，主成分矩阵

        返回:
        - updated_A_class_features: dict，每个类别的更新后的 A 类特征
        """
        updated_A_class_features = {}

        for class_idx, A_features in A_class_features.items():
            if len(A_features) > 0:
                A_features_flattened = A_features.view(-1, A_features.size(-1))
                W_class = W[class_idx]
                projected_features = A_features_flattened @ W_class
                updated_A_class_features[class_idx] = projected_features @ W_class.T
            else:
                updated_A_class_features[class_idx] = A_features
        return updated_A_class_features

    def reconstruct_and_classify(self, B_class_features, W):
        """
        对 B 类特征进行主成分重构，并根据重构误差来分类
        """
        reconstructed_B_features = {key: [] for key in B_class_features}
        updated_B_class_features = {}
        # new_B_class = {i: [[] for j in range(len(B_class_features[i]))] for i in range(self.num_classes)}
        for class_idx, B_features in B_class_features.items():
            if len(B_features) > 0:
                B_features_flattened = B_features.view(-1, B_features.size(-1))  # (N_class, D)
                W_class = W[class_idx]  # (D, num_components)
                projected_features = B_features_flattened @ W_class  # (N_class, num_components)
                updated_B_class_features[class_idx] = projected_features @ W_class.T
            else:
                reconstructed_B_features[class_idx] = B_features

        return reconstructed_B_features

    def combine_predictions(self, updated_A_features, reconstruct_B_features, A_class_positions, B_class_positions,
                            feature_shape):
        """
        根据位置信息恢复特征图。

        参数:
        - updated_A_features: dict，A 类更新后的特征，每个类别为 key，值为特征张量 (N_class, D)
        - reconstruct_B_features: dict，B 类重构后的特征，每个类别为 key，值为特征张量 (N_class, D)
        - A_class_positions: dict，A 类特征的位置，每个类别为 key，值为位置信息 (N_class, 3)
        - B_class_positions: dict，B 类特征的位置，每个类别为 key，值为位置信息 (N_class, 3)
        - feature_shape: tuple，特征图的形状 (B, D, H, W)

        返回:
        - final_features: torch.Tensor，恢复后的特征图，大小为 (B, D, H, W)
        """
        if len(updated_A_features.shape) == 4:
            B, D, H, W = feature_shape
            BHW = B * H * W
        else:
            B, D, H, W, Z = feature_shape
            BHW = B * H * W * Z


        # 初始化恢复后的特征图
        if len(updated_A_features.shape) == 4:
            final_features = torch.zeros(B, D, H, W, device='cuda')
        else:
            final_features = torch.zeros(B, D, H, W, Z, device='cuda')
        final_features_ = final_features.reshape(D, BHW)
        for class_idx, features in updated_A_features.items():
            if len(features)>0:
                positions = A_class_positions[class_idx]  # A 类的扁平化编号 (N_class,)
                final_features_[:, positions] = features.unsqueeze(0).T

        for class_idx, features in reconstruct_B_features.items():
            if len(features) > 0:
                positions = B_class_positions[class_idx]  # B 类的扁平化编号 (N_class,)
                final_features_[:, positions] = features.unsqueeze(0).T
        if len(updated_A_features.shape) == 4:
            final_features = final_features_.view(D, B, H, W).permute(1, 0, 2, 3)
        else:
            final_features = final_features_.view(D, B, H, W, Z).permute(1, 0, 2, 3, 4)

        return final_features
