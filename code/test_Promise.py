import argparse
import os
import shutil
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib import pyplot as plt
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from nets.net_factory import net_factory, BCP_net
# from nets.net_factory_pcaflow import net_factory, BCP_net

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data_split/Promise', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, asd

def test_single_volume(case, net, test_save_path):
    np_data_path = os.path.join(FLAGS.root_path+ '/Promise12/npy_image')
    img = np.load(os.path.join(np_data_path, '{}.npy'.format(case)))
    mask = np.load(os.path.join(np_data_path, '{}_segmentation.npy'.format(case)))
    prediction = np.zeros_like(mask)
    for ind in range(img.shape[0]):
        slice = img[ind, :, :]
        label_slice = mask[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        label_slice = torch.from_numpy(label_slice)
        net.eval()
        with torch.no_grad():
            # print(net(input)[0].shape)2
            out = torch.argmax(torch.softmax(net(input)[0], dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out

        if False:
            # out = label_slice
            fig, ax = plt.subplots()
            # 旋转和翻转底图与叠加图
            transformed_slice = np.flip(np.rot90(slice, 3), axis=1)
            transformed_out = np.flip(np.rot90(out, 3), axis=1)
            # 将 out 中像素值为 0 的部分屏蔽掉
            masked_out = np.ma.masked_where(transformed_out == 0, transformed_out)

            # 先显示底图，再显示叠加图（不使用 colormap）
            ax.imshow(transformed_slice, cmap='gray')
            ax.imshow(masked_out, cmap='jet')  # 默认按照原始数据显示，不指定 cmap
            ax.axis('off')
            # plt.show()
            # 保存图像并移除白边
            img_path = test_save_path + case +str(ind)
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
    if np.sum(prediction == 1) == 0:
        first_metric = 0 ,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, mask == 1)
    return first_metric

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])

    snapshot_path = "./model/BCP/Promise_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = "./model/BCP/Promise_{}_{}_labeled/{}_predictions/".format(FLAGS.exp,
                                                                                                             FLAGS.labelnum,
                                                                                                             FLAGS.model)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model,in_chns=1,class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric = test_single_volume(case, net, test_save_path)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list)]
    return avg_metric, test_save_path

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))