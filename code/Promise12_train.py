import argparse
import logging
import os
import random
import shutil
import sys
from skimage.measure import label
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch
from medpy import metric
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataloaders.dataset import (TwoStreamBatchSampler,RandomGenerator)
from dataloaders.promise12 import Promise12
from nets.net_factory_pcaflow import BCP_net, net_factory
from utils import losses, ramps, val_2d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data_split/Promise', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CPP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seg_rate', type=float, default=1/4)
parser.add_argument('--lp', type=float,  default=0.8, help='linear_prob rate')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
#flx
parser.add_argument('--weightedbce', type=bool, default=True, help='bce with weights') # 相对置信度函数，对训练的影响
parser.add_argument('--alpha', type=float, default=4.5, help='bce with weights') # 相对置信度函数参数
parser.add_argument('--beta', type=float, default=2.7, help='bce with weights') # 相对置信度函数参数
parser.add_argument('--BG', type=bool, default=False, help='Background') # 处理全是背景类的情况
parser.add_argument('--pre_epoch', type=int, default=50, help='Background')
parser.add_argument('--self_epoch', type=int, default=100, help='Background')
parser.add_argument('--pca_weight', type=int, default=0.05, help='Background')
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=2)
def to_one_hot(labels, num_classes):
    # one_hot 后返回形状为 (B, H, W, C)
    labels = labels.type(torch.int64)
    one_hot_labels = F.one_hot(labels, num_classes=num_classes)
    one_hot_labels = one_hot_labels.permute(0, 3, 1, 2)
    return one_hot_labels
def test_single_volume_promise(image, label, net, classes):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = net(input)
            if len(out)>1:
                out = out[0]
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_promise(prediction == i, label == i))
    return metric_list
def calculate_metric_percase_promise(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    return dice

def generate_mask(img,seg_rate):
    # img(6,1,256,256)
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * seg_rate), int(img_y * seg_rate)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0 # (6,256,256)
    return mask.long(), loss_mask.long()
def get_Mix_label(back,fore,mask):
    back,fore = back.type(torch.int64), fore.type(torch.int64)
    mix_label = back * mask + fore * (1-mask)
    return mix_label

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, mode=0, BGweights=1):
    if args.weightedbce:
        CE = losses.WeightedCrossEntropyLoss()
    else:
        CE = nn.CrossEntropyLoss(reduction='none')

    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if not isinstance(BGweights, int):
        BGweights = BGweights.view(len(BGweights), 1, 1)
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    if mode == 1 and args.weightedbce:
        weights = losses.bce_weight2(output_soft,args.alpha,args.beta)
        loss_ce = image_weight * (CE(output, img_l, weights) * mask).sum() / (mask.sum() + 1e-16)
        loss_ce += patch_weight * (CE(output, patch_l) * patch_mask * BGweights).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    elif mode == 2 and args.weightedbce:
        weights = losses.bce_weight2(output_soft,args.alpha,args.beta)
        loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
        loss_ce += patch_weight * (CE(output, patch_l, weights) * patch_mask * BGweights).sum() / (
                    patch_mask.sum() + 1e-16)  # loss = loss_ce
    else:
        loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
        loss_ce += patch_weight * (CE(output, patch_l) * patch_mask * BGweights).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce
def mix_loss2(output,plabel):
    CE= nn.CrossEntropyLoss()
    plabel = plabel.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    Loss_dice = dice_loss(output_soft, plabel.unsqueeze(1))
    Loss_ce = CE(output, plabel)
    return Loss_dice,Loss_ce


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])
def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []  # 保存每个样本处理结果
    N = segmentation.shape[0]  # batch size
    for i in range(0, N):
        temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i]) 获取当前样本的分割结果
        temp_prob = torch.zeros_like(temp_seg)  # 创造全0张量存储类别掩码
        temp_prob[temp_seg == 1] = 1  # 设置当前类别c的区域为1
        temp_prob = temp_prob.detach().cpu().numpy()  # 转为numpy数组
        labels = label(temp_prob)  # 使用scipy的连通区域标记算法获取连通区域
        if labels.max() != 0:  # 如果存在连通区域
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1  # 标签中出现次数最多的连通区域，即最大的连通区域
        else:
            largestCC=temp_prob # 没有连通区域，则使用原始temp_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()

# 从模型输出中提取二值化的mask
def get_ACDC_masks(output, nms=0):
    # 对网络输出应用 softmax，得到每个像素属于不同类的概率分布
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1) # 选取每个像素的最大类别，得到每个像素的所属类别
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs) # 提取每个类最大连通区域
    return probs
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Promise":
        ref_dict = {"7": 299, "3": 130}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)
    ce_loss = CrossEntropyLoss()

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode = "train")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Promise12(base_dir=args.root_path, split='train', out_size=256,
                         transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = Promise12(base_dir=args.root_path, split='val', out_size=256)
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labeled_num)  # args.labeled_num=7
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    W = 0
    W_list = []
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a,args.seg_rate)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            if epoch > args.pre_epoch:
                 out_mixl, pcaW, loss_pca = model(net_input, W)
            else:
                 out_mixl, pcaW, loss_pca = model(net_input)

            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss = (1 - args.pca_weight) * ((loss_dice + loss_ce) / 2) + args.pca_weight * loss_pca

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))


            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_promise(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i], iter_num)
                performance = np.mean(metric_list, axis=0)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
            if epoch <= args.pre_epoch:
                W_list.append(pcaW.detach() )
                W = torch.mean(torch.stack(W_list, dim=0),dim=0)
            else:
                W = W + (1/iter_num) * (pcaW.detach() - W)

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_train(args, pre_snapshot_path,snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode = "train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode = "train", ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Promise12(base_dir=args.root_path, split='train', out_size=256,
                         transform=transforms.Compose([RandomGenerator(args.patch_size)])
                         )
    db_val = Promise12(base_dir=args.root_path, split='val', out_size=256)
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labeled_num)  # args.labeled_num=7
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()
    ce_loss = CrossEntropyLoss()
    W = 0
    W_list = []
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance= 0.0
    iterator = tqdm(range(max_epoch), ncols=70)  #
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                               args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[
                                                                                              args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                # pre_a, eno_a, deo_a = ema_model(uimg_a)
                # pre_b, eno_b, deo_b = ema_model(uimg_b)
                pre_a, pcaW_a, loss_pca_a = ema_model(uimg_a)
                pre_b, pcaW_a, loss_pca_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                img_mask, loss_mask = generate_mask(img_a,args.seg_rate)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            # out_unl, unenfea, undefea = model(net_input_unl)
            # out_l, enfea, defea = model(net_input_l)
            if epoch > args.self_epoch:
                out_unl, pcaW_unl, loss_pca_unl = model(net_input_unl, W)
                out_l, pcaW_l, loss_pca_l = model(net_input_l, W)
            else:
                out_unl, pcaW_unl, loss_pca_unl = model(net_input_unl)
                out_l, pcaW_l, loss_pca_l = model(net_input_l)


            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True, mode=1)
            l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight, mode=2)

            loss_ce = unl_ce + l_ce
            loss_dice = unl_dice + l_dice

            paper_loss = ((loss_dice + loss_ce) / 2)
            pca_loss = ((loss_pca_unl + loss_pca_l) / 2)

            loss = (1 - args.pca_weight) * paper_loss + args.pca_weight * pca_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (
            iter_num, loss, loss_dice, loss_ce))
            # logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f,pc_ce: %f' % (
            #     iter_num, loss, loss_dice, loss_ce, loss_pc))
            if iter_num % 20 == 0:
                image = net_input_unl[1, 0:1, :, :]  # 从混合（无标签背景+有标签前景）样本中 取第一个batch的第一个通道图像
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)  # 它的预测结果
                labs = unl_label[1, ...].unsqueeze(0) * 50  # 放大50倍，清晰可见
                writer.add_image('train/Un_GroundTruth', labs, iter_num)  # 它的gt

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_promise(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                writer.add_scalar('info/model_val_{}_dice'.format(1), metric_list, iter_num)
                performance = np.mean(metric_list, axis=0)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                print(performance)
                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break

            pcaW = (pcaW_l + pcaW_unl) / 2

            if epoch <= args.self_epoch:
                W_list.append(pcaW.detach())
                W = torch.mean(torch.stack(W_list, dim=0), dim=0)
            else:
                W = W + (1 / iter_num) * (pcaW.detach() - W)

        if iter_num >= max_iterations:
            iterator.close()
            break
        writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "./model/BCP/Promise_{}_{}_labeled/pre_train".format(args.exp, args.labeled_num)
    self_snapshot_path = "./model/BCP/Promise_{}_{}_labeled/self_train".format(args.exp, args.labeled_num)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Promise12_train.py'), self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
