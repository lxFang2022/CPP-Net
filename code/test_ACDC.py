import argparse
import os
import shutil
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm   
import matplotlib.pyplot as plt
from nets.net_factory_pcaflow import net_factory, BCP_net
# from nets.net_factory import net_factory, BCP_net
from sklearn.manifold import TSNE
import torch.nn.functional as F
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data_split/ACDC', help='Name of Experiment')
# parser.add_argument('--exp', type=str, default='BCP_MM_20', help='experiment_name')
parser.add_argument('--exp', type=str, default='BCP_TMI6', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')
# parser.add_argument('--last', type=str, default='iter_24000_dice_0.8854', help='best or last')
parser.add_argument('--last', type=str, default=None, help='best or last')


def tsne_plot(feat_chw: torch.Tensor, label_hw: torch.Tensor,id,img_path,
              per_class=1000, perplexity=30, seed=42):
    """
    feat_chw: [C, H, W]   特征图 (decoder最后一层 16通道 或 logits 4通道)
    label_hw: [H, W]      对应像素的类别标签 (int)
    """
    C, H, W = feat_chw.shape
    X = feat_chw.permute(1,2,0).reshape(-1, C).cpu().numpy()   # [H*W, C]
    y = label_hw.reshape(-1)                   # [H*W]

    # 按类别均衡抽样，避免某类过多
    rng = np.random.default_rng(seed)
    idxs = []
    for c in np.unique(y):
        ids = np.where(y == c)[0]
        take = ids if len(ids) <= per_class else rng.choice(ids, per_class, replace=False)
        idxs.append(take)
    idx = np.concatenate(idxs)
    X, y = X[idx], y[idx]

    # 标准化
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, X.shape[0]//3),
                init="pca", learning_rate=200, n_iter=1000, random_state=seed, verbose=1)
    Z = tsne.fit_transform(X)

    palette = sns.color_palette("Set2", n_colors=len(set(y)))

    plt.figure(figsize=(6, 6))
    for cls in sorted(set(y)):
        idx = (y == cls)
        plt.scatter(Z[idx, 0], Z[idx, 1],
                    s=12, alpha=0.7, marker='o',
                    color=palette[cls], label=f"Class {cls}", edgecolors='none')

    plt.xticks([]);
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(img_path+"tsne_features_"+str(id)+".png", dpi=300, bbox_inches='tight')
    plt.show()

    return Z, y


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def _minmax_norm(x: np.ndarray, eps=1e-6):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)



def _overlay_binary_mask(gray_hw: np.ndarray,
                         mask_hw: np.ndarray,
                         color=(1.0, 0.0, 0.0),
                         alpha: float = 0.5):
    """
    灰度图上叠加一个二值 mask，用于画高置信度区域。
    """
    H, W = gray_hw.shape
    base = np.stack([gray_hw, gray_hw, gray_hw], axis=-1).astype(np.float32)
    layer = np.zeros((H, W, 3), dtype=np.float32)
    m = mask_hw.astype(bool)
    layer[m] = np.array(color, dtype=np.float32)
    out = (1 - alpha) * base + alpha * layer
    return np.clip(out, 0.0, 1.0)

from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt

def _overlay_seg(gray_hw: np.ndarray,
                 seg_hw: np.ndarray,
                 palette=None,
                 alpha=0.6,
                 draw_edge=True,
                 edge_thick_iter: int = 1):
    """
    灰度图 + 语义分割标签叠加，可视化 GT 边界等。
    gray_hw: (H,W) float in [0,1]
    seg_hw:  (H,W) int, 背景=0
    edge_thick_iter: 边界膨胀迭代次数，1 即加粗一倍
    """
    H, W = gray_hw.shape
    base = np.stack([gray_hw, gray_hw, gray_hw], axis=-1).astype(np.float32)

    if palette is None:
        palette = {
            1: (1.0, 1.0, 0.0),  # yellow
            2: (0.0, 1.0, 0.0),  # green
            3: (1.0, 0.0, 1.0),  # magenta
        }

    out = base.copy()
    seg = seg_hw.astype(np.int32)

    # 区域填充
    for c, color in palette.items():
        if c == 0:
            continue
        mask = (seg == c)
        if not mask.any():
            continue
        layer = np.zeros((H, W, 3), dtype=np.float32)
        layer[mask] = np.array(color, dtype=np.float32)
        out = (1 - alpha) * out + alpha * layer

    # 边界绘制 + 加粗
    if draw_edge and find_boundaries is not None:
        for c, color in palette.items():
            if c == 0:
                continue
            edge = find_boundaries(seg == c, mode='outer')
            edge_thick = binary_dilation(edge, iterations=edge_thick_iter)
            out[edge_thick] = np.array(color, dtype=np.float32)

    return np.clip(out, 0.0, 1.0)


def _get_zoom_roi(mask_hw: np.ndarray,
                  H: int,
                  W: int,
                  crop: int = 64):
    """
    根据 GT 前景 mask 自动计算一个放大区域的坐标 [y0:y1, x0:x1]。
    如果前景为空，则以图像中心为放大区域。
    """
    ys, xs = np.where(mask_hw > 0)
    if len(xs) == 0:
        cy, cx = H // 2, W // 2
    else:
        cy = (ys.min() + ys.max()) // 2
        cx = (xs.min() + xs.max()) // 2

    half = crop // 2
    y0 = max(0, cy - half)
    y1 = min(H, cy + half)
    x0 = max(0, cx - half)
    x1 = min(W, cx + half)

    # 保证区域尺寸尽量为 crop×crop
    if (y1 - y0) < crop:
        if y0 == 0:
            y1 = min(H, crop)
        elif y1 == H:
            y0 = max(0, H - crop)
    if (x1 - x0) < crop:
        if x0 == 0:
            x1 = min(W, crop)
        elif x1 == W:
            x0 = max(0, W - crop)

    return y0, y1, x0, x1

def visualize_anchor_student_1x4(
        base_img,
        gt_mask,
        teacher_prob,
        high_conf_mask,
        teacher_edge,
        student_edge,
        save_path,
        palette=None,
        zoom_crop: int = 64,
        roi=None):   # 新增参数 roi=(y0,y1,x0,x1)

    if palette is None:
        palette = {
            1: (1.0, 1.0, 0.0),  # yellow
            2: (0.0, 1.0, 0.0),  # green
            3: (1.0, 0.0, 1.0),  # magenta
        }

    img = base_img  # (H,W)
    H, W = img.shape

    # ====== 先生成四幅大图的数据 ======
    # (a) 原图 + GT 叠加（带粗边）
    overlay_a = _overlay_seg(img, gt_mask, palette=palette, alpha=0.6,
                             draw_edge=True, edge_thick_iter=1)

    # (b) 教师 max prob
    if teacher_prob.ndim == 3:   # (C,H,W)
        heatmap = teacher_prob.max(0)
    else:                        # (H,W)
        heatmap = teacher_prob
    heatmap = np.asarray(heatmap)

    # (c) 高置信度区域 mask 覆盖在原图上
    overlay_c = np.stack([img, img, img], axis=-1).astype(np.float32)
    overlay_c[high_conf_mask] = np.array([0.6, 0.0, 0.0], dtype=np.float32)  # 红色高置信区域

    # (d) 教师 vs 学生 轮廓对比（边缘加粗）
    overlay_d = np.stack([img, img, img], axis=-1).astype(np.float32)
    teacher_edge_thick = binary_dilation(teacher_edge, iterations=1)
    student_edge_thick = binary_dilation(student_edge, iterations=1)
    overlay_d[teacher_edge_thick] = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 教师：绿
    overlay_d[student_edge_thick] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 学生：红

    # ====== 计算放大区域 ROI ======
    if roi is None:
        # 原来的自动计算
        y0, y1, x0, x1 = _get_zoom_roi(gt_mask, H, W, crop=zoom_crop)
    else:
        # 手动指定
        y0, y1, x0, x1 = roi

    # ====== 正式画图 ======
    fig, axes = plt.subplots(
        1, 4,
        figsize=(12, 3),
        dpi=300,
        gridspec_kw={'wspace': 0.02}
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.axis('off')

    # ---- (a) ----
    axes[0].imshow(overlay_a, origin='lower')
    rect_a = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                           edgecolor='white', facecolor='none', linewidth=1.2)
    axes[0].add_patch(rect_a)
    axins0 = inset_axes(axes[0], width="35%", height="35%", loc="upper right",
                        borderpad=0.5)
    axins0.imshow(overlay_a[y0:y1, x0:x1], origin='lower')
    axins0.set_xticks([]); axins0.set_yticks([])
    axins0.set_aspect('equal')
    # ⭐ 放大图也画白色框（用轴的边框实现）
    for spine in axins0.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('white')
        spine.set_linewidth(1.2)

    # ---- (b) ----
    axes[1].imshow(heatmap, cmap='viridis', vmin=0.0, vmax=1.0, origin='lower')
    rect_b = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                           edgecolor='white', facecolor='none', linewidth=1.2)
    axes[1].add_patch(rect_b)
    axins1 = inset_axes(axes[1], width="35%", height="35%", loc="upper right",
                        borderpad=0.5)
    axins1.imshow(heatmap[y0:y1, x0:x1], cmap='viridis', vmin=0.0, vmax=1.0, origin='lower')
    axins1.set_xticks([]); axins1.set_yticks([])
    axins1.set_aspect('equal')
    for spine in axins1.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('white')
        spine.set_linewidth(1.2)

    # ---- (c) ----
    axes[2].imshow(overlay_c, origin='lower')
    rect_c = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                           edgecolor='white', facecolor='none', linewidth=1.2)
    axes[2].add_patch(rect_c)
    axins2 = inset_axes(axes[2], width="35%", height="35%", loc="upper right",
                        borderpad=0.5)
    axins2.imshow(overlay_c[y0:y1, x0:x1], origin='lower')
    axins2.set_xticks([]); axins2.set_yticks([])
    axins2.set_aspect('equal')
    for spine in axins2.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('white')
        spine.set_linewidth(1.2)

    # ---- (d) ----
    axes[3].imshow(overlay_d, origin='lower')
    rect_d = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                           edgecolor='white', facecolor='none', linewidth=1.2)
    axes[3].add_patch(rect_d)
    axins3 = inset_axes(axes[3], width="35%", height="35%", loc="upper right",
                        borderpad=0.5)
    axins3.imshow(overlay_d[y0:y1, x0:x1], origin='lower')
    axins3.set_xticks([]); axins3.set_yticks([])
    axins3.set_aspect('equal')
    for spine in axins3.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('white')
        spine.set_linewidth(1.2)

    plt.subplots_adjust(
        left=0.01, right=0.99,
        bottom=0.01, top=0.99,
        wspace=0.02, hspace=0.0
    )

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_TSNE_2(out_class, mode, fea):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances
    from matplotlib.gridspec import GridSpec
    # out_class = out_class.detach().cpu()
    mode = mode.detach().cpu()
    fea = fea.detach().cpu()
    # ======================
    # 1. 数据预处理（使用您的实际数据）
    # ======================
    # 假设已有变量：
    # - out_class : 256x256       (dtype=int)
    # - fea       : 16x256x256    (dtype=float)
    # - mode      : 4x16x3        (dtype=float)

    # 展平特征和类别数据
    features_flattened = fea.reshape(16, -1).T  # 形状 (65536, 16)
    categories_flattened = out_class.ravel()  # 形状 (65536,)

    # 标准化特征
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_flattened)

    # ======================
    # 可视化1：特征空间降维
    # ======================
    # 使用PCA降维
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_normalized)

    # 投影模式特征（标准化后转换）
    mode_flat = mode.reshape(-1, 16)  # 形状 (12, 16)
    mode_normalized = scaler.transform(mode_flat)
    modes_2d = pca.transform(mode_normalized).reshape(4, 3, 2)

    # 绘制散点图（下采样10%避免内存问题）
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[::10, 0], features_2d[::10, 1],
                          c=categories_flattened[::10],
                          s=1, alpha=0.3, cmap='tab10')

    # 标注模式特征
    markers = ['*', 'o', 's']  # 不同模式的标记
    for cls in range(4):
        for mode_idx in range(3):
            x, y = modes_2d[cls, mode_idx]
            plt.scatter(x, y, s=150, marker=markers[mode_idx],
                        edgecolor='black', label=f'Class {cls} Mode {mode_idx + 1}')

    plt.colorbar(scatter, label='Class ID')
    plt.title("PCA Projection of Features with Mode Markers")
    plt.xlabel("PC1"), plt.ylabel("PC2")
    plt.legend(ncol=4, loc='upper right', bbox_to_anchor=(1.0, -0.1))
    plt.tight_layout()
    plt.show()

    # ======================
    # 可视化2：模式匹配热图
    # ======================
    def compute_match_map(category, features, mode_features):
        """计算每个像素与所属类模式的最大余弦相似度"""
        h, w = category.shape
        match_heatmap = np.zeros((h, w))

        for cls in range(4):
            mask = (category == cls)
            if not np.any(mask): continue

            # 提取当前类别的特征 (N,16)
            cls_features = features.squeeze()[:, mask].T  # 转换为 (N,16)

            # 计算与3个模式的余弦相似度
            similarities = []
            for mode_idx in range(3):
                mode_vec = mode_features[cls, :, mode_idx]
                norm = np.linalg.norm(cls_features, axis=1) * np.linalg.norm(mode_vec)
                sim = np.dot(cls_features, mode_vec) / (norm + 1e-8)
                similarities.append(sim)

            # 取最大相似度
            max_sim = np.max(similarities, axis=0)
            match_heatmap[mask] = max_sim

        return match_heatmap

    # 生成热图
    match_heatmap = compute_match_map(out_class, fea, mode)

    # 绘制热图与类别叠加
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(out_class, cmap='tab10')
    plt.title("Class Segmentation Map")

    plt.subplot(122)
    plt.imshow(match_heatmap, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title("Mode Matching Heatmap")
    plt.tight_layout()
    plt.show()
    # ======================
    # 可视化3：通道响应可视化
    # ======================
    # 选择方差最大的3个通道
    # channel_var = np.var(fea, axis=(1, 2))
    # top3_channels = np.argsort(channel_var)[-3:][::-1]
    #
    # # 绘制通道响应
    # plt.figure(figsize=(15, 5))
    # for i, ch in enumerate(top3_channels):
    #     plt.subplot(1, 3, i + 1)
    #     plt.imshow(fea[ch], cmap='hot')
    #     plt.colorbar()
    #     plt.title(f"Channel {ch} (Variance={channel_var[ch]:.2f})")
    # plt.tight_layout()

    # ======================
    # 可视化4：类别-模式距离分布
    # ======================
    plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2)

    for cls in range(4):
        cls_mask = (categories_flattened == cls)
        if np.sum(cls_mask) == 0: continue
        cls_features = features_normalized[cls_mask]

        # 提取当前类的三个模式
        cls_modes = mode_normalized[cls * 3: (cls + 1) * 3]  # 从标准化后的模式中获取

        # 计算欧氏距离
        distances = pairwise_distances(cls_features, cls_modes, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        # 绘制直方图
        ax = plt.subplot(gs[cls // 2, cls % 2])
        ax.hist(min_distances, bins=50, density=True, alpha=0.7)
        ax.set_title(f"Class {cls} Distance Distribution")
        ax.set_xlabel("Min Distance to Modes")

    plt.tight_layout()
    plt.show()

def plot_TSNE(mode_features):

    num_classes = 4
    num_modes = 3
    feature_dim = 16

    # # 将 shape 从 (4, 16, 3) 变为 (4×3, 16)，即 (12, 16)
    # features = mode_features.permute(0, 2, 1).reshape(num_classes * num_modes, feature_dim).detach().cpu()
    #
    # # 执行 t-SNE 进行降维
    # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    # features_2d = tsne.fit_transform(features.numpy())
    #
    # # 设置不同类别的颜色
    # colors = ['red', 'blue', 'green', 'purple']
    # markers = ['o', 's', '^']  # 每个类别的3个模式特征使用不同的 marker
    #
    # # 可视化
    # plt.figure(figsize=(8, 6))
    # for class_idx in range(num_classes):
    #     for mode_idx in range(num_modes):
    #         idx = class_idx * num_modes + mode_idx
    #         plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
    #                     color=colors[class_idx], marker=markers[mode_idx],
    #                     label=f'Class {class_idx + 1} - Mode {mode_idx + 1}' if mode_idx == 0 else None,
    #                     edgecolors='black')
    #
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.title('t-SNE Visualization of Mode Features')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # 重新调整形状，使其变为 (12, 16)，即 (模式数 × 类别数, 特征维度)
    # features = mode_features.reshape(num_modes * num_classes, feature_dim).detach().cpu()
    #
    # # 执行 t-SNE 进行降维
    # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    # features_2d = tsne.fit_transform(features.numpy())
    #
    # # 设置不同模式的颜色（3 种模式）
    # colors = ['red', 'blue', 'green']
    #
    # # 设置不同类别的 marker（4 种类别）
    # markers = ['o', 's', '^', 'D']  # 圆圈、方块、三角、菱形
    #
    # # 可视化
    # plt.figure(figsize=(8, 6))
    # for mode_idx in range(num_modes):
    #     for class_idx in range(num_classes):
    #         idx = mode_idx * num_classes + class_idx
    #         plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
    #                     color=colors[mode_idx], marker=markers[class_idx],
    #                     label=f'Mode {mode_idx + 1} - Class {class_idx + 1}' if class_idx == 0 else None,
    #                     edgecolors='black')
    #
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.title('t-SNE Visualization of 3 Modes x 4 Classes')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # 颜色：3 个模式（Mode 1, 2, 3）
    colors = ['red', 'blue', 'green']

    # 形状（Marker）：4 个类别（Class 1, 2, 3, 4）
    markers = ['o', 's', '^', 'D']

    # 创建子图，每个类别单独一张图
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 4))

    for class_idx in range(num_classes):
        # 取出该类别的 3 个模式特征 (3, 16)
        features = mode_features[class_idx].detach().cpu().numpy()  # shape: (3, 16)

        # t-SNE 降维 (3, 16) -> (3, 2)
        tsne = TSNE(n_components=2, perplexity=3, random_state=42)
        features_2d = tsne.fit_transform(features)

        # 绘制当前类别的模式特征
        ax = axes[class_idx]
        for mode_idx in range(num_modes):
            ax.scatter(features_2d[mode_idx, 0], features_2d[mode_idx, 1],
                       color=colors[mode_idx], marker=markers[class_idx],
                       label=f'Mode {mode_idx + 1}', edgecolors='black')

        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title(f'Class {class_idx + 1}')
        ax.legend()
        ax.grid(True)

    plt.suptitle('t-SNE Visualization of Mode Features for Each Class')
    plt.tight_layout()
    plt.show()

def test_single_volume(case, net_student, test_save_path, FLAGS,
                       tau=0.7):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]   # [D,H,W]
    label = h5f['label'][:]   # [D,H,W]
    prediction = np.zeros_like(label)

    # 当前 case 的可视化目录
    case_viz_dir = test_save_path
    os.makedirs(case_viz_dir, exist_ok=True)

    D = image.shape[0]
    mid_idx = D // 2    # 取中间那一层索引用来可视化

    for ind in range(D):
        slice_img   = image[ind, ...]    # (H,W)
        slice_label = label[ind, ...]    # (H,W)
        x, y = slice_img.shape

        # ---------- resize 到网络输入 ----------
        slice_resized = zoom(slice_img,   (256 / x, 256 / y), order=0)
        label_resized = zoom(slice_label, (256 / x, 256 / y), order=0)
        inp = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()

        # ---------- 教师 & 学生预测 ----------
        net_student.eval()
        with torch.no_grad():

            out_s = net_student(inp)
            if isinstance(out_s, (list, tuple)):
                out_s = out_s[0]
            pred_s = torch.argmax(torch.softmax(out_s, dim=1), dim=1).squeeze(0)
            pred_s = pred_s.cpu().numpy()                           # (256,256)

        # ---------- 还原回原分辨率做 metric ----------
        pred_orig = zoom(pred_s, (x / 256, y / 256), order=0)
        prediction[ind] = pred_orig

    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = "./model/BCP/ACDC_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    # net = BCP_net(in_chns=1, class_num=FLAGS.num_classes)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    if FLAGS.last:
        save_model_path = os.path.join(snapshot_path, '{}.pth'.format(FLAGS.last))
    else:
        save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path))

    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for i, case in enumerate(tqdm(image_list)):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS
        )
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
