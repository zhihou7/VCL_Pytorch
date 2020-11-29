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
from networks.ResNet50_HICO_torch import HICO_HOI
from ult.timer import Timer
import numpy as np
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
                        default='res101', type=str)
    args = parser.parse_args()

    return args

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
    test_generator = obtain_test_dataset1(args.object_thres, args.human_thres,
                         stride=stride, test_type=args.test_type, model_name=args.model)
    import torch
    model = HICO_HOI(args.model)
    checkpoint = torch.load(weight + '/' + '{}_checkpoint.pth.tar'.format(args.iteration))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # epoch = checkpoint['epoch']
    # mean_best = checkpoint['mean_best']
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    detection = {}
    count = 0
    _t = {'im_detect': Timer(), 'misc': Timer()}
    for image, blobs, image_id in test_generator:
        _t['im_detect'].tic()
        image = image.transpose([0, 3, 1, 2])
        blobs['sp'] = blobs['sp'].transpose([0, 3, 1, 2]).astype(np.float32)

        model(torch.from_numpy(image).to(device), image_id, blobs['H_num'],
              torch.from_numpy(blobs['H_boxes']).to(device),
              torch.from_numpy(blobs['O_boxes']).to(device), None,
              torch.from_numpy(blobs['sp']).to(device), False)
        sp = model.predictions["cls_prob_sp"]
        hoi = model.predictions["cls_prob_hoi"]
        spHOI = torch.mul(sp, hoi)
        import ipdb

        #
        sp = sp.detach().cpu().numpy()
        hoi = hoi.detach().cpu().numpy()
        spHOI = spHOI.detach().cpu().numpy()

        last_img_id = -1
        # ipdb.set_trace()
        print(len(blobs['H_boxes']), )
        temp = [[blobs['H_boxes'][i], blobs['O_boxes'][i], blobs['O_cls'][i],
                 0, blobs['H_score'][i], blobs['O_score'][i], 0, 0, sp[i], hoi[i],
                 spHOI[i]] for i in range(len(blobs['H_boxes']))]
        if image_id in detection:
            detection[image_id].extend(temp)
        else:
            detection[image_id] = temp

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(model.model_name, count, 10566, image_id,
                                                                       _t['im_detect'].average_time), end='',
              flush=True)

    # TODO remove
    # pickle.dump(detection, open('test_new.pkl', 'wb'))
    pickle.dump(detection, open(output_file, "wb"))
