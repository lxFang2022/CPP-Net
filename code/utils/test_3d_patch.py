import h5py
import math
import os
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def getLargestCC(segmentation):
    labels = label(segmentation)
    # assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else:
        largestCC = segmentation
    return largestCC


def var_all_case_LA(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, root_path='./data_split/LA'):
    with open(os.path.join(root_path, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = [os.path.join(root_path, "2018LA_Seg_Training Set", item.strip(), "mri_norm2.h5")
                  for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice
def dice_per_case(pred, gt, num_classes=16):
    """返回当前 case 的平均 Dice 及逐类 Dice 列表（背景 0 忽略）"""
    per_cls = []
    for c in range(1, num_classes):               # 跳过背景
        pred_c, gt_c = (pred == c), (gt == c)
        if pred_c.sum() == 0 and gt_c.sum() == 0:  # 同时缺失 → 完全正确
            per_cls.append(np.nan)
        else:
            try:
                per_cls.append(metric.binary.dc(pred_c, gt_c))
            except RuntimeError:                   # 点太少时可能报错
                per_cls.append(np.nan)
    return np.nanmean(per_cls), per_cls           # 宏平均 Dice
def var_all_case_AMOS(model, num_classes=16,
                      patch_size=(112,112,80), stride_xy=18, stride_z=4, root_path="./data/amos_data"):
    path = os.path.join(root_path, "val")
    image_list = sorted([f.replace('_image.npy','')
                         for f in os.listdir(path) if f.endswith('_image.npy')])

    total_macro_dice = 0.0
    all_per_cls = [[] for _ in range(num_classes-1)]   # 汇总每类 Dice

    for case in tqdm(image_list, desc="Validate"):
        img = np.load(os.path.join(path, f"{case}_image.npy"))
        gt  = np.load(os.path.join(path, f"{case}_label.npy"))

        # 把 80 维度移到最后
        img = np.moveaxis(img, np.where(np.array(img.shape)==80)[0][0], -1)
        gt  = np.moveaxis(gt,  np.where(np.array(gt.shape )==80)[0][0], -1)

        pred, _ = test_single_case_AMOS(
            model, img, stride_xy, stride_z, patch_size, num_classes=num_classes
        )

        macro_dice, per_cls = dice_per_case(pred, gt, num_classes)
        total_macro_dice += macro_dice
        for i, d in enumerate(per_cls):
            all_per_cls[i].append(d)

    avg_macro_dice = total_macro_dice / len(image_list)
    avg_per_cls = [np.nanmean(c) for c in all_per_cls]

    print(f"\n=== Validation Result ===")
    print(f"Macro Dice (mean over classes) : {avg_macro_dice:.4f}")
    for idx, d in enumerate(avg_per_cls, 1):
        print(f"  Class {idx:02d}: {d:.4f}")

    return avg_macro_dice
def var_all_case_Parse(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, root_path='./data_split/Parse2022'):
    base_dir = root_path
    list_path = os.path.join(base_dir, 'test.txt')
    with open(list_path, 'r') as f:
        case_id = [line.strip() for line in f.readlines()]

    image_dir = os.path.join(base_dir, 'image')
    label_dir = os.path.join(base_dir, 'label')

    image_paths = [os.path.join(image_dir, f"{id}.nii.gz") for id in case_id]
    label_paths = [os.path.join(label_dir, f"{id}.nii.gz") for id in case_id]

    total_dice = 0.0
    for image_path, label_path in tqdm(zip(image_paths, label_paths), total=len(case_id)):
        # 加载NIfTI文件
        try:
            image = nib.load(image_path).get_fdata().astype(np.float32)
            label = nib.load(label_path).get_fdata().astype(np.uint8)

            # 测试单个病例
            prediction, _ = test_single_case(
                model, image, stride_xy, stride_z, patch_size, num_classes=num_classes
            )

            # 计算Dice系数
            dice = 0 if np.sum(prediction) == 0 else metric.binary.dc(prediction, label)
            total_dice += dice

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    avg_dice = total_dice / len(case_id)
    print(f'Average Dice score: {avg_dice:.4f}')
    return avg_dice


def var_all_case_ImageCAS(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, root_path='./data_split/ImageCAS'):
    base_dir = root_path
    list_path = os.path.join(base_dir, f'test.txt')
    with open(list_path, 'r') as f:
        case_id = [line.strip() for line in f.readlines()]

    image_dir = os.path.join(base_dir, 'image')
    label_dir = os.path.join(base_dir, 'label')
    file_suffix = '.nii.gz'  # 根据实际文件命名修改
    image_paths = os.path.join(image_dir, f'{case_id}.nii.gz')
    label_paths = os.path.join(label_dir, f'{case_id}{file_suffix}')

    total_dice = 0.0
    for image_path, label_path in tqdm(zip(image_paths, label_paths), total=len(case_id)):
        # 加载NIfTI文件
        try:
            image = nib.load(image_path).get_fdata().astype(np.float32)
            label = nib.load(label_path).get_fdata().astype(np.uint8)

            # 添加batch和channel维度 (1×1×D×H×W)
            image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)

            # 测试单个病例
            prediction, _ = test_single_case(
                model, image, stride_xy, stride_z, patch_size, num_classes=num_classes
            )

            # 计算Dice系数
            dice = 0 if np.sum(prediction) == 0 else metric.binary.dc(prediction, label)
            total_dice += dice

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    avg_dice = total_dice / len(case_id)
    print(f'Average Dice score: {avg_dice:.4f}')
    return avg_dice


def test_all_case(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])

        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
            ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)

        if False:
            label = prediction.astype(np.float32)
            verts, faces, normals, values = measure.marching_cubes(label, level=0.7)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            # 创建等值面的 3D 多边形集合
            mesh = Poly3DCollection(verts[faces], alpha=0.9)
            mesh.set_facecolor([1, 0, 0])
            mesh.set_edgecolor('black')
            mesh.set_linewidths(0.05)
            ax.add_collection3d(mesh)
            # 设置坐标轴范围
            ax.set_xlim(0, label.shape[0])
            ax.set_ylim(0, label.shape[1])
            ax.set_zlim(0, label.shape[2])
            ax.view_init(elev=270, azim=180)
            ax.set_axis_off()

            plt.tight_layout()
            # plt.show()

            img_path = test_save_path + str(ith)
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_pred.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = model(test_patch)
                    if len(y1) > 1:
                        y1 = y1[0]
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, 1, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int64)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map

def test_single_case_AMOS(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    # ---------- 滑窗推理 ----------
    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])

                patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                patch = torch.from_numpy(patch[None, None].astype(np.float32)).cuda()  # (1,1,D,H,W)

                with torch.no_grad():
                    out = model(patch)
                    if isinstance(out, (list, tuple)):  # (B,C,D,H,W) in out[0]
                        out = out[0]
                    prob = F.softmax(out, dim=1)  # (1,16,D,H,W)

                prob_np = prob.cpu().numpy()[0]  # (16,D,H,W)
                score_map[:, xs:xs + patch_size[0],
                ys:ys + patch_size[1],
                zs:zs + patch_size[2]] += prob_np

                cnt[xs:xs + patch_size[0],
                ys:ys + patch_size[1],
                zs:zs + patch_size[2]] += 1

    # ---------- 累积平均 ----------
    score_map = score_map / np.expand_dims(cnt, axis=0)

    # ---------- 取最大概率类别 ----------
    label_map = np.argmax(score_map, axis=0).astype(np.int64)  # shape (ww,hh,dd)

    # ---------- 去掉推理 pad ----------
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                    hl_pad:hl_pad + h,
                    dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w,
                    hl_pad:hl_pad + h,
                    dl_pad:dl_pad + d]

    return label_map, score_map

def var_all_case_LA_plus(model_l, model_r, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, root_path='./data_split/LA'):
    with open(os.path.join(root_path, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = [os.path.join(root_path, "2018LA_Seg_Training Set", item.strip(), "mri_norm2.h5")
                  for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size,
                                                      num_classes=num_classes)
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def test_all_case_plus(model_l, model_r, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                       save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size,
                                                      num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])

        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
            ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_pred.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric


def test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1_l, _ = model_l(test_patch)
                    y1_r, _ = model_r(test_patch)
                    y1 = (y1_l + y1_r) / 2
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, 1, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
