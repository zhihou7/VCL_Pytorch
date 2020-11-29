# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import _init_paths
from torch.utils.data import Dataset, DataLoader

from networks.ResNet50_HICO_torch import HICO_HOI
from ult.timer import Timer

os.environ['DATASET'] = 'HICO'

import numpy as np
import argparse
import pickle
import ipdb

from ult.config import cfg
from ult.ult import obtain_data, get_zero_shot_type, get_augment_type, generator2

import torch
import random

seed = 10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def _init_fn(worker_id):
    np.random.seed(int(seed))


class HicoDataset(Dataset):

    def __init__(self, Pos_augment=15, Neg_select=60, augment_type=0, with_pose=False, zero_shot_type=0,
                 large_neg_for_ho=False, isalign=False, epoch=0):

        if with_pose:
            Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose.pkl', "rb"),
                                      encoding='latin1')
            Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_pose.pkl', "rb"),
                                     encoding='latin1')
        else:
            Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"), encoding='latin1')
            Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb"), encoding='latin1')

        g = generator2

        if with_pose:
            pattern_channel = 3
        else:
            pattern_channel = 2
        from functools import partial
        self.generator = generator2(Trainval_GT, Trainval_N, Pos_augment, Neg_select,
                                    augment_type, with_pose, zero_shot_type, isalign, epoch)

    def __len__(self):
        return 800000

    def __getitem__(self, idx):
        im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern = next(self.generator)
        im_orig1, image_id1, num_pos1, Human_augmented1, Object_augmented1, action_HO1, Pattern1 = next(self.generator)

        return im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern, \
               im_orig1, image_id1, num_pos1, Human_augmented1, Object_augmented1, action_HO1, Pattern1



def parse_args():
    parser = argparse.ArgumentParser(description='Train VCL on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
                        help='Number of iterations to perform',
                        default=1500010, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='VCL_union_l2_rew_aug5_3_x5new_res101', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
                        help='Number of augmented detection for each one. (By jittering the object detections)',
                        default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
                        help='Number of Negative example selected for each image',
                        default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
                        help='How many ResNet blocks are there?',
                        default=5, type=int)
    args = parser.parse_args()
    return args


def cal_vcl_loss(model, fc7_O, fc7_V, new_gt_class_HO, device):

    fc7_vo = model.head_to_tail_ho(fc7_O, fc7_V, None, None, True, 'fc_HO')
    model.region_classification_ho(fc7_vo, True, None,
                                      'classification', nameprefix='merge_')
    cls_score_hoi = model.predictions["merge_cls_score_hoi"]
    reweights = model.HO_weight.to(device)


    # this means that we also apply the reweight strategy for the generated HO relation
    # TODO I simply and empirically set the weights for VCL. I think there should be a better solution.
    #  Noticeably, our method is orthogonal to re-weighting strategy.
    #  Moreover, completely copying from previous work, we multiply the weights at the logits.
    #  I think this is also an important reason why baseline of zero-shot has some values!
    #  This can help the network learn from the known factors (i.e. verb and object)
    #  It might be because the non-linear sigmoid function.
    #  After this kind of re-weighting, the small value (e.g. 0.1) will further tend 0 where the gradient
    #  is larger. It is interesting! We do not mention this in paper since our method is orthogonal to this.
    #  But I do not understand the reason very good. Hope someone can explain.
    cls_score_hoi = torch.mul(cls_score_hoi, reweights / 10)

    return model.loss(new_gt_class_HO, cls_score_hoi)


if __name__ == '__main__':

    args = parse_args()
    print(args)
    args.model = args.model.strip()

    Trainval_GT = None
    Trainval_N = None

    np.random.seed(cfg.RNG_SEED)
    if args.model.__contains__('res101'):
        weight = cfg.ROOT_DIR + '/Weights/res101_faster_rcnn_iter_1190000.ckpt'
    else:
        weight = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    print(weight)
    tb_dir = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.LOCAL_DATA + '/Weights/' + args.model + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.Restore_flag == 5:
        if os.path.exists(output_dir + 'checkpoint'):
            args.Restore_flag = -1
        elif args.model.__contains__('unique_weights'):
            args.Restore_flag = 6

    augment_type = get_augment_type(args.model)

    model = HICO_HOI(args.model)

    with_pose = False
    # if args.model.__contains__('pose'):
    #     with_pose = True

    coco = False
    zero_shot_type = get_zero_shot_type(args.model)
    large_neg_for_ho = False

    dataset = HicoDataset(Pos_augment=args.Pos_augment,
                          Neg_select=args.Neg_select,
                          augment_type=augment_type,
                          with_pose=with_pose,
                          zero_shot_type=zero_shot_type,
                          )
    dataloader_train = DataLoader(dataset, 1,
                                  shuffle=False, num_workers=2,
                                  worker_init_fn=_init_fn)  # num_workers=batch_size
    trainables = []
    not_trainables = []
    for name, p in model.named_parameters():

        if name.split('.')[0] == 'Conv_pretrain' or name.__contains__('base_model.1') \
                or name.__contains__('base_model.5') or name.__contains__('base_model.6')\
                or name.__contains__('base_model.7'):
            p.requires_grad = False
            not_trainables.append(p)
            print('not train', name)
        else:
            print('train', name)
            p.requires_grad= True
            trainables.append(p)


    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()


    model.apply(set_bn_eval)

    print(model)
    import torch.optim as optim

    optimizer = optim.SGD(params=trainables, lr=cfg.TRAIN.LEARNING_RATE * 10,
                          momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # lambda1 = lambda epoch: 1.0 if epoch < 10 else (10 if epoch < 28 else 1)
    lambda1 = lambda epoch: 1.0 if epoch < 10 else (0.1 if epoch < 28 else 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    device = torch.device("cuda")
    model.to(device)
    timer = Timer()
    # (im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern)
    i = 0

    for item in dataloader_train:

        timer.tic()

        O_features = []
        V_features = []
        gt_class_HOI = []
        num_stop_list = []
        tower_losses = []

        step_size = int(cfg.TRAIN.STEPSIZE * 5)
        if (i+1) % step_size == 0:
            scheduler.step()

        im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern, \
        im_orig1, image_id1, num_pos1, Human_augmented1, Object_augmented1, action_HO1, Pattern1 = item

        im_orig = im_orig.to(device)
        num_pos = num_pos.to(device)
        Human_augmented = Human_augmented.to(device)
        Object_augmented = Object_augmented.to(device)
        action_HO = action_HO.to(device)
        Pattern = Pattern.to(device)

        im_orig1 = im_orig1.to(device)
        num_pos1 = num_pos1.to(device)
        Human_augmented1 = Human_augmented1.to(device)
        Object_augmented1 = Object_augmented1.to(device)
        action_HO1 = action_HO1.to(device)
        Pattern1 = Pattern1.to(device)

        optimizer.zero_grad()
        # print(im_orig.shape, Human_augmented.shape)
        model(im_orig[0], image_id[0], num_pos[0], Human_augmented[0], Object_augmented[0], action_HO[0], Pattern[0],
              True)
        num_stop = model.get_num_stop(num_pos[0], Human_augmented[0])
        model.add_loss(action_HO[0], num_stop, device)
        comp_num_stop = model.get_compose_num_stop(num_pos[0], Human_augmented[0])
        O_features.append(model.intermediate['fc7_O'][:comp_num_stop])
        V_features.append(model.intermediate['fc7_verbs'][:comp_num_stop])
        num_stop_list.append(comp_num_stop)
        tower_losses.append(model.losses['total_loss'])
        gt_class_HOI.append(action_HO[0])

        model(im_orig1[0], image_id1[0], num_pos1[0], Human_augmented1[0], Object_augmented1[0], action_HO1[0], Pattern1[0],
              True)
        num_stop = model.get_num_stop(num_pos1[0], Human_augmented1[0])
        model.add_loss(action_HO1[0], num_stop, device)
        comp_num_stop = model.get_compose_num_stop(num_pos1[0], Human_augmented1[0])
        O_features.append(model.intermediate['fc7_O'][:comp_num_stop])
        V_features.append(model.intermediate['fc7_verbs'][:comp_num_stop])
        num_stop_list.append(comp_num_stop)
        tower_losses.append(model.losses['total_loss'])
        gt_class_HOI.append(action_HO1[0])

        if not model.model_name.__contains__('_base'):
            O_feats = torch.cat(O_features, dim=0)
            V_feats = torch.cat(V_features[::-1], dim=0)

            hoi_labels = torch.cat(gt_class_HOI, dim=0)
            o_hois = torch.matmul(torch.matmul(hoi_labels, model.obj_to_HO_matrix.transpose()).type(torch.bool).type(torch.float32),
                                  model.obj_to_HO_matrix)
            v_hois = torch.matmul(torch.matmul(hoi_labels, model.verb_to_HO_matrix.transpose()).type(torch.bool).type(torch.float32),
                                  model.verb_to_HO_matrix)
            v_hois = torch.cat([v_hois[len(V_features[0]):], v_hois[:len(V_features[1])]], dim=0)
            composite_hoi_label = torch.logical_and(o_hois.type(torch.bool), v_hois.type(torch.bool)).type(torch.float32)

            vcl_loss = cal_vcl_loss(model, O_feats, V_feats, composite_hoi_label, device)
            tower_losses.append(vcl_loss * 0.5)
        else:
            pass

        final_loss = torch.sum(tower_losses)
        final_loss.backward()

        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(p, 1.)
        optimizer.step()
        i += 1
        timer.toc()
        if i % (cfg.TRAIN.SNAPSHOT_ITERS * 5) == 0 or i == 10:
            torch.save({
                'iteration': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}, output_dir +'{}_checkpoint.pth.tar'.format(i))
        if i % 500 == 0:
            print('\rstep {} sp: {} hoi: {} total: {} lr: {} speed: {:.3f} s/iter \r'.format(i, model.losses['sp_cross_entropy'].item(),
                                                                      model.losses['hoi_cross_entropy'].item(),
                                                                      model.losses['total_loss'].item(),
                                                                      scheduler.get_lr(), timer.average_time))
            torch.cuda.empty_cache()