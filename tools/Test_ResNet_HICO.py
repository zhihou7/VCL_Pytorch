# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import _init_paths
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms, ToTensor

from networks.ResNet50_HICO_torch import HICO_HOI
from ult.timer import Timer

os.environ['DATASET'] = 'HICO'

import argparse


from ult.config import cfg
from models.test_HICO import obtain_test_dataset1, test_net_data_api1

def parse_args():
    parser = argparse.ArgumentParser(description='Test VCL on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=10, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='VCL_union_l2_rew_aug5_3_x5new_res101', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.3, type=float) 
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)
    # TODO For better object detector, the object_thres and human_thres should also be changed accordingly.
    #  e.g. in our fine-tuned detector, object_thres and human_thres is 0.1 and 0.3 respectively.
    parser.add_argument('--debug', dest='debug',
                        help='Human threshold',
                        default=0, type=int)
    parser.add_argument('--type', dest='test_type',
                        help='Human threshold',
                        default='vcl', type=str)
    args = parser.parse_args()

    return args


class HicoDataset(Dataset):

    def __init__(self, Pos_augment=15, Neg_select=60, augment_type=0, with_pose=False, zero_shot_type=0,
                 large_neg_for_ho=False, isalign=False, epoch=0, transform=None):
        self.transform = transform

        self.generator = obtain_test_dataset1(args.object_thres, args.human_thres,
                         stride=stride, test_type=args.test_type, model_name=args.model)

    def __len__(self):
        return 800000

    def __getitem__(self, idx):
        # im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern = next(self.generator)
        im_orig, blobs, image_id  = next(self.generator)
        # im_orig = im_orig.transpose([0, 3, 1, 2])
        # Pattern = Pattern.transpose([0, 3, 1, 2]).astype(np.float32)
        # Human_augmented = Human_augmented.astype(np.float32)
        # Object_augmented = Object_augmented.astype(np.float32)
        # Human_augmented = Human_augmented.astype(np.float32)
        # print(im_orig.dtype, Pattern.dtype)
        # print(im_orig)
        # print(im_orig.shape, im_orig)
        blobs['sp'] = blobs['sp'].transpose([0, 3, 1, 2]).astype(np.float32)
        if self.transform:
            im_orig = self.transform(im_orig)
        # print('after', im_orig)
        print(np.asarray(im_orig).shape)
        return im_orig, image_id, blobs


if __name__ == '__main__':

    args = parse_args()
    print(args)
    # test detections result
    from sys import version_info

    weight = cfg.ROOT_DIR + '/Weights/' + args.model

    import os

    print('weight:', weight)
    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight )
    output_file = cfg.LOCAL_DATA + '/Results/' + str(args.iteration) + '_' + args.model + '_tin.pkl'

    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'
    stride = 200
    # test_generator = obtain_test_dataset1(args.object_thres, args.human_thres,
    #                      stride=stride, test_type=args.test_type, model_name=args.model)
    import torch
    model = HICO_HOI(args.model)
    checkpoint = torch.load(weight + '/' + '{}_checkpoint.pth.tar'.format(args.iteration))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    for name, p in model.named_parameters():
        print(name, p.mean())

    dataset = HicoDataset(
                          transform=transforms.Compose([ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])]),
                          )
    dataloader_test = DataLoader(dataset, 1,
                                  shuffle=False, num_workers=1)
    # transform = transforms.Compose([ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                  std=[0.229, 0.224, 0.225])])
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    detection = {}
    count = 0
    last_update_value = {}
    _t = {'im_detect': Timer(), 'misc': Timer()}
    for image, image_id, blobs in dataloader_test:
        _t['im_detect'].tic()
        print(image.shape, image.mean(), image.std())
        print(image.shape, image.mean(), image.std())

        model(image.to(device), image_id[0], blobs['H_num'][0],
              blobs['H_boxes'][0].to(device),
              blobs['O_boxes'][0].to(device), None,
              blobs['sp'][0].to(device), False)
        sp = model.predictions["cls_prob_sp"]
        hoi = model.predictions["cls_prob_hoi"]
        spHOI = torch.mul(sp, hoi)
        import ipdb
        ipdb.set_trace()
        #
        sp = sp.detach().cpu().numpy()
        hoi = hoi.detach().cpu().numpy()

        # print([h[:10] for h in hoi])
        # print([h[:10] for h in sp])
        # print(image.mean(), image.std())



        spHOI = spHOI.detach().cpu().numpy()

        if count == 10:
            for k in model.state_dict().keys():
                tmp = model.state_dict()[k].type(torch.float32).mean().detach().cpu().numpy()
                if k in last_update_value:
                    print(tmp)
                    if abs(last_update_value[k] - tmp) > 0:
                        print(k, last_update_value[k], tmp, last_update_value[k] - tmp)

                last_update_value[k] = tmp
                # print(k, model.state_dict()[k].type(torch.float32).mean())
            print('-'*80)


        last_img_id = -1
        # ipdb.set_trace()
        # print(len(blobs['H_boxes']), )
        temp = [[blobs['H_boxes'][i][1:], blobs['O_boxes'][i][1:], blobs['O_cls'][i],
                 0, blobs['H_score'][i], blobs['O_score'][i], 0, 0, sp[i], hoi[i],
                 0] for i in range(len(blobs['H_boxes']))]
        if image_id in detection:
            detection[image_id].extend(temp)
        else:
            detection[image_id] = temp

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(model.model_name, count, 10566, image_id,
                                                                       _t['im_detect'].average_time), end='',
              flush=True)
        # if count > 100:
        #     break

    # TODO remove
    # pickle.dump(detection, open('test_new.pkl', 'wb'))
    pickle.dump(detection, open(output_file, "wb"))
