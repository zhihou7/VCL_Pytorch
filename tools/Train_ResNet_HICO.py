# --------------------------------------------------------

# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import _init_paths
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, ToTensor

from networks.ResNet50_HICO_torch import HICO_HOI
from ult.timer import Timer


import numpy as np
import argparse
import pickle
import ipdb

from ult.config import cfg
from ult.ult import obtain_data, get_zero_shot_type, get_augment_type, generator2

import torch
import random

# seed = 10
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(seed)
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


def _init_fn(worker_id):
    # np.random.seed(int(seed))
    pass


class HicoDataset(Dataset):

    def __init__(self, Pos_augment=15, Neg_select=60, augment_type=0, with_pose=False, zero_shot_type=0,
                 large_neg_for_ho=False, isalign=False, epoch=0, transform=None):


        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"), encoding='latin1')
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb"), encoding='latin1')

        self.transform = transform
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
        # im_orig = im_orig.transpose([0, 3, 1, 2])
        # Pattern = Pattern.transpose([0, 3, 1, 2]).astype(np.float32)
        # Human_augmented = Human_augmented.astype(np.float32)
        # Object_augmented = Object_augmented.astype(np.float32)
        # Human_augmented = Human_augmented.astype(np.float32)
        # print(im_orig.dtype, Pattern.dtype)
        # print(im_orig)
        # print(im_orig.shape, im_orig)
        if self.transform:
            im_orig = self.transform(im_orig[0])
        # print('after', im_orig)
        return im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern


def parse_args():
    parser = argparse.ArgumentParser(description='Train VCL on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
                        help='Number of iterations to perform',
                        default=200000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='VCL_humans_aug5_3_x5new_res101_1', type=str)
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


if __name__ == '__main__':

    args = parse_args()
    print(args)
    args.model = args.model.strip()

    Trainval_GT = None
    Trainval_N = None
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
                          transform=transforms.Compose([ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]),
                          )
    dataloader_train = DataLoader(dataset, 1,
                                  shuffle=False, num_workers=1,
                                  worker_init_fn=_init_fn)  # num_workers=batch_size
    trainables = []
    not_trainables = []
    for name, p in model.named_parameters():

        if name.__contains__('base_model.0') or name.__contains__('base_model.1') \
                or name.__contains__('base_model.4') or name.__contains__('bn')\
                or name.__contains__('HOI_MLP.1') or name.__contains__('sp_MLP.1')\
                or name.__contains__('HOI_MLP.5') or name.__contains__('sp_MLP.5')\
                or name.__contains__('downsample.1'):
            #BN
            p.requires_grad = False
            not_trainables.append(p)
            print('not train', name, p.mean(), p.std())

        else:
            print('train', name, p.mean(), p.std())
            p.requires_grad= True
            trainables.append(p)


    def set_bn_eval(m):
        classname = m.__class__.__name__
        # print(m)
        if classname.find('BatchNorm') != -1:
            m.eval()
            # print(m, '======')


    model.apply(set_bn_eval)
    # exit()
    print(model)
    import torch.optim as optim

    optimizer = optim.SGD(params=trainables, lr=cfg.TRAIN.LEARNING_RATE * 10,
                          momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # lambda1 = lambda epoch: 1.0 if epoch < 10 else (10 if epoch < 28 else 1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.TRAIN.GAMMA)

    device = torch.device("cuda")
    model.to(device)
    timer = Timer()
    # (im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern)
    i = 0
    last_update_value = {}
    for item in dataloader_train:
    # for item in dataset:
        im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern = item

        if len(Human_augmented[0]) <= 1 or num_pos[0] <= 1:
            continue
        timer.tic()
        step_size = int(cfg.TRAIN.STEPSIZE * 5)
        if (i+1) % step_size == 0:
            scheduler.step()


        im_orig = im_orig.to(device)
        num_pos = num_pos.to(device)
        Human_augmented = Human_augmented.to(device)
        Object_augmented = Object_augmented.to(device)
        action_HO = action_HO.to(device)
        Pattern = Pattern.to(device)
        optimizer.zero_grad()
        # print(im_orig.shape, Human_augmented.shape)
        # print(im_orig[0].mean(), im_orig[0].std(), image_id, Human_augmented[0], len(Object_augmented[0]), len(action_HO[0]), len(Pattern[0]))
        model(im_orig, image_id[0], num_pos[0], Human_augmented[0], Object_augmented[0], action_HO[0], Pattern[0],
              True)
        num_stop = model.get_num_stop(num_pos[0], Human_augmented[0])
        model.add_loss(action_HO[0], num_stop, device)
        model.losses['total_loss'].backward()
        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(p, 1.)
        optimizer.step()
        i += 1
        # for name, p in model.named_parameters():
        #     print(name, p.mean())
        # import ipdb;ipdb.set_trace()
        # if i == 10 or i == 1000:
        #
        #     for k in model.state_dict().keys():
        #         tmp = model.state_dict()[k].type(torch.float32).mean().detach().cpu().numpy()
        #         if k in last_update_value:
        #
        #             if abs(last_update_value[k] - tmp) > 0:
        #                 print(k, last_update_value[k], tmp, last_update_value[k] - tmp)
        #
        #         last_update_value[k] = tmp
        #         # print(k, model.state_dict()[k].type(torch.float32).mean())
        #     print(im_orig.mean(), im_orig.std())
        #     print('-'*80)
        # exit()
        # print(model.state_dict().keys())
        # print(model.state_dict());exit()
        timer.toc()
        if i % (cfg.TRAIN.SNAPSHOT_ITERS * 5) == 0 or i == 10 or i == 1000:
            torch.save({
                'iteration': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}, output_dir +'{}_checkpoint.pth.tar'.format(i))

        if i % 500 == 0 or i < 10:
            print('\rstep {} img id: {} sp: {} hoi: {} total: {} lr: {} speed: {:.3f} s/iter \r'.format(i, image_id[0], model.losses['sp_cross_entropy'].item(),
                                                                      model.losses['hoi_cross_entropy'].item(),
                                                                      model.losses['total_loss'].item(),
                                                                      scheduler.get_lr(), timer.average_time))
            torch.cuda.empty_cache()