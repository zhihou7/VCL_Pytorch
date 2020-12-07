# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Transferable-Interactiveness-Network, Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim import arg_scope
# from tensorflow.contrib.slim.python.slim.nets import resnet_utils
# from tensorflow.contrib.slim.python.slim.nets import resnet_v1
# from tensorflow.python.framework import ops

from ult.tools import get_convert_matrix
from ult.config import cfg
from ult.visualization import draw_bounding_boxes_HOI
import torch
import numpy as np


import torch.nn as nn
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.size()[0], -1)
        # return x.view(x.size()[0], -1)


class HICO_HOI(nn.Module):
    def __init__(self, model_name):
        super(HICO_HOI, self).__init__()
        import torchvision.models as models


        # model.eval()
        # pred = model([img])

        self.model_name = model_name
        self.visualize = {}
        self.test_visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}

        # self.image       =   None # tf.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image')
        # self.spatial     = None # tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name = 'sp')
        # self.H_boxes     = None # tf.placeholder(tf.float32, shape=[None, 5], name = 'H_boxes')
        # self.O_boxes     = None # tf.placeholder(tf.float32, shape=[None, 5], name = 'O_boxes')
        # gt_class_HO = None # tf.placeholder(tf.float32, shape=[None, 600], name = 'gt_class_HO')
        # self.H_num       = None # tf.placeholder(tf.int32)    # positive nums
        # self.image_id    = None # tf.placeholder(tf.int32)
        self.num_classes = 600
        self.compose_num_classes = 600
        self.num_fc      = 1024
        self.verb_num_classes = 117
        self.obj_num_classes = 80
        self.scope       = 'resnet_v1_50'
        self.stride      = [16, ]
        # if tf.__version__ == '1.1.0':
        #     raise Exception('wrong tensorflow version 1.1.0')
        # else:
            # from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            # self.blocks = [resnet_v1_block('block1', base_depth=64,  num_units=3, stride=2),
            #                resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
            #                resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
            #                resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
            #                resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]
            # if self.model_name.__contains__('unique_weights') or self.model_name.__contains__('_pa3')\
            #         or self.model_name.__contains__('_pa4'):
            #     print("add block6 unique_weights2")
            #     self.blocks.append(resnet_v1_block('block6', base_depth=512, num_units=3, stride=1))

        """We copy from TIN. calculated by log(1/(n_c/sum(n_c)) c is the category and n_c is the number of samples"""
        self.HO_weight = np.array([
            9.192927, 9.778443, 10.338059, 9.164914, 9.075144, 10.045923, 8.714437, 8.59822, 12.977117, 6.2745423,
            11.227917, 6.765012, 9.436157, 9.56762, 11.0675745, 11.530198, 9.609821, 9.897503, 6.664475, 6.811699,
            6.644726, 9.170454, 13.670264, 3.903943, 10.556748, 8.814335, 9.519224, 12.753973, 11.590822, 8.278912,
            5.5245695, 9.7286825, 8.997436, 10.699849, 9.601237, 11.965516, 9.192927, 10.220277, 6.056692, 7.734048,
            8.42324, 6.586457, 6.969533, 10.579222, 13.670264, 4.4531965, 9.326459, 9.288238, 8.071842, 10.431585,
            12.417501, 11.530198, 11.227917, 4.0678477, 8.854023, 12.571651, 8.225684, 10.996116, 11.0675745,
            10.100731,
            7.0376034, 7.463688, 12.571651, 14.363411, 5.4902234, 11.0675745, 14.363411, 8.45805, 10.269067,
            9.820116,
            14.363411, 11.272368, 11.105314, 7.981595, 9.198626, 3.3284247, 14.363411, 12.977117, 9.300817,
            10.032678,
            12.571651, 10.114916, 10.471591, 13.264799, 14.363411, 8.01953, 10.412168, 9.644913, 9.981384,
            7.2197933,
            14.363411, 3.1178555, 11.031207, 8.934066, 7.546675, 6.386472, 12.060826, 8.862153, 9.799063, 12.753973,
            12.753973, 10.412168, 10.8976755, 10.471591, 12.571651, 9.519224, 6.207762, 12.753973, 6.60636,
            6.2896967,
            4.5198326, 9.7887, 13.670264, 11.878505, 11.965516, 8.576513, 11.105314, 9.192927, 11.47304, 11.367679,
            9.275815, 11.367679, 9.944571, 11.590822, 10.451388, 9.511381, 11.144535, 13.264799, 5.888291,
            11.227917,
            10.779892, 7.643191, 11.105314, 9.414651, 11.965516, 14.363411, 12.28397, 9.909063, 8.94731, 7.0330057,
            8.129001, 7.2817025, 9.874775, 9.758241, 11.105314, 5.0690055, 7.4768796, 10.129305, 9.54313, 13.264799,
            9.699972, 11.878505, 8.260853, 7.1437693, 6.9321113, 6.990665, 8.8104515, 11.655361, 13.264799,
            4.515912,
            9.897503, 11.418972, 8.113436, 8.795067, 10.236277, 12.753973, 14.363411, 9.352776, 12.417501,
            0.6271591,
            12.060826, 12.060826, 12.166186, 5.2946343, 11.318889, 9.8308115, 8.016022, 9.198626, 10.8976755,
            13.670264,
            11.105314, 14.363411, 9.653881, 9.503599, 12.753973, 5.80546, 9.653881, 9.592727, 12.977117, 13.670264,
            7.995224, 8.639826, 12.28397, 6.586876, 10.929424, 13.264799, 8.94731, 6.1026597, 12.417501, 11.47304,
            10.451388, 8.95624, 10.996116, 11.144535, 11.031207, 13.670264, 13.670264, 6.397866, 7.513285, 9.981384,
            11.367679, 11.590822, 7.4348736, 4.415428, 12.166186, 8.573451, 12.977117, 9.609821, 8.601359, 9.055143,
            11.965516, 11.105314, 13.264799, 5.8201604, 10.451388, 9.944571, 7.7855496, 14.363411, 8.5463,
            13.670264,
            7.9288645, 5.7561946, 9.075144, 9.0701065, 5.6871653, 11.318889, 10.252538, 9.758241, 9.407584,
            13.670264,
            8.570397, 9.326459, 7.488179, 11.798462, 9.897503, 6.7530537, 4.7828183, 9.519224, 7.6492405, 8.031909,
            7.8180614, 4.451856, 10.045923, 10.83705, 13.264799, 13.670264, 4.5245686, 14.363411, 10.556748,
            10.556748,
            14.363411, 13.670264, 14.363411, 8.037262, 8.59197, 9.738439, 8.652985, 10.045923, 9.400566, 10.9622135,
            11.965516, 10.032678, 5.9017305, 9.738439, 12.977117, 11.105314, 10.725825, 9.080208, 11.272368,
            14.363411,
            14.363411, 13.264799, 6.9279733, 9.153925, 8.075553, 9.126969, 14.363411, 8.903826, 9.488214, 5.4571533,
            10.129305, 10.579222, 12.571651, 11.965516, 6.237189, 9.428937, 9.618479, 8.620408, 11.590822,
            11.655361,
            9.968962, 10.8080635, 10.431585, 14.363411, 3.796231, 12.060826, 10.302968, 9.551227, 8.75394,
            10.579222,
            9.944571, 14.363411, 6.272396, 10.625742, 9.690582, 13.670264, 11.798462, 13.670264, 11.724354,
            9.993963,
            8.230013, 9.100721, 10.374427, 7.865129, 6.514087, 14.363411, 11.031207, 11.655361, 12.166186, 7.419324,
            9.421769, 9.653881, 10.996116, 12.571651, 13.670264, 5.912144, 9.7887, 8.585759, 8.272101, 11.530198,
            8.886948,
            5.9870906, 9.269661, 11.878505, 11.227917, 13.670264, 8.339964, 7.6763024, 10.471591, 10.451388,
            13.670264,
            11.185357, 10.032678, 9.313555, 12.571651, 3.993144, 9.379805, 9.609821, 14.363411, 9.709451, 8.965248,
            10.451388, 7.0609145, 10.579222, 13.264799, 10.49221, 8.978916, 7.124196, 10.602211, 8.9743395, 7.77862,
            8.073695, 9.644913, 9.339531, 8.272101, 4.794418, 9.016304, 8.012526, 10.674532, 14.363411, 7.995224,
            12.753973, 5.5157638, 8.934066, 10.779892, 7.930471, 11.724354, 8.85808, 5.9025764, 14.363411,
            12.753973,
            12.417501, 8.59197, 10.513264, 10.338059, 14.363411, 7.7079706, 14.363411, 13.264799, 13.264799,
            10.752493,
            14.363411, 14.363411, 13.264799, 12.417501, 13.670264, 6.5661197, 12.977117, 11.798462, 9.968962,
            12.753973,
            11.47304, 11.227917, 7.6763024, 10.779892, 11.185357, 14.363411, 7.369478, 14.363411, 9.944571,
            10.779892,
            10.471591, 9.54313, 9.148476, 10.285873, 10.412168, 12.753973, 14.363411, 6.0308623, 13.670264,
            10.725825,
            12.977117, 11.272368, 7.663911, 9.137665, 10.236277, 13.264799, 6.715625, 10.9622135, 14.363411,
            13.264799,
            9.575919, 9.080208, 11.878505, 7.1863923, 9.366199, 8.854023, 9.874775, 8.2857685, 13.670264, 11.878505,
            12.166186, 7.616999, 9.44343, 8.288065, 8.8104515, 8.347254, 7.4738197, 10.302968, 6.936267, 11.272368,
            7.058223, 5.0138307, 12.753973, 10.173757, 9.863602, 11.318889, 9.54313, 10.996116, 12.753973,
            7.8339925,
            7.569945, 7.4427395, 5.560738, 12.753973, 10.725825, 10.252538, 9.307165, 8.491293, 7.9161053,
            7.8849015,
            7.782772, 6.3088884, 8.866243, 9.8308115, 14.363411, 10.8976755, 5.908519, 10.269067, 9.176025,
            9.852551,
            9.488214, 8.90809, 8.537411, 9.653881, 8.662968, 11.965516, 10.143904, 14.363411, 14.363411, 9.407584,
            5.281472, 11.272368, 12.060826, 14.363411, 7.4135547, 8.920994, 9.618479, 8.891141, 14.363411,
            12.060826,
            11.965516, 10.9622135, 10.9622135, 14.363411, 5.658909, 8.934066, 12.571651, 8.614018, 11.655361,
            13.264799,
            10.996116, 13.670264, 8.965248, 9.326459, 11.144535, 14.363411, 6.0517673, 10.513264, 8.7430105,
            10.338059,
            13.264799, 6.878481, 9.065094, 8.87035, 14.363411, 9.92076, 6.5872955, 10.32036, 14.363411, 9.944571,
            11.798462, 10.9622135, 11.031207, 7.652888, 4.334878, 13.670264, 13.670264, 14.363411, 10.725825,
            12.417501,
            14.363411, 13.264799, 11.655361, 10.338059, 13.264799, 12.753973, 8.206432, 8.916674, 8.59509,
            14.363411,
            7.376845, 11.798462, 11.530198, 11.318889, 11.185357, 5.0664344, 11.185357, 9.372978, 10.471591,
            9.6629305,
            11.367679, 8.73579, 9.080208, 11.724354, 5.04781, 7.3777695, 7.065643, 12.571651, 11.724354, 12.166186,
            12.166186, 7.215852, 4.374113, 11.655361, 11.530198, 14.363411, 6.4993753, 11.031207, 8.344818,
            10.513264,
            10.032678, 14.363411, 14.363411, 4.5873594, 12.28397, 13.670264, 12.977117, 10.032678, 9.609821
        ], dtype='float32').reshape(1, 600)
        self.HO_weight = torch.from_numpy(self.HO_weight)
        num_inst_path = cfg.ROOT_DIR +  '/Data/num_inst.npy'
        num_inst = np.load(num_inst_path)
        self.num_inst = num_inst

        verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix(self.verb_num_classes, self.obj_num_classes)

        self.obj_to_HO_matrix = obj_to_HO_matrix
        self.verb_to_HO_matrix = verb_to_HO_matrix
        # self.gt_obj_class = torch.matmul(gt_class_HO, self.obj_to_HO_matrix.transpose())
        # self.gt_verb_class = torch.matmul(gt_class_HO, self.verb_to_HO_matrix.transpose())


        import torchvision
        if model_name.__contains__('fpn'):
            fpn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            fpn_model = fpn_model.backbone.body
            self.base_model = nn.Sequential(*list(fpn_model.children())[:-1])
        else:
            fpn_model = models.resnet50(pretrained=True)
            self.base_model = nn.Sequential(*list(fpn_model.children())[0:7])
        self.flat = Flatten()
        # import ipdb
        # ipdb.set_trace()



        self.h_block = fpn_model.layer4
        # print(self.h_block)
        import copy
        self.o_block = copy.deepcopy(self.h_block) # get a new instance
        # print(self.o_block,)
        # self.o_block.load_state_dict(self.h_block.state_dict()) # copy weights and stuff
        # self.v_block = fpn_model.layer4

        self.Conv_sp = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(5, 5), stride=(1, 1), bias=True),
            # nn.BatchNorm2d(64, eps=1e-05, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            # nn.AdaptiveMaxPool2d([2, 2]),
            nn.MaxPool2d([2, 2]),
            nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), bias=True),
            # nn.BatchNorm2d(32, eps=1e-05, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d([2, 2]),
            # nn.ReLU(inplace=False),
            Flatten(),
        )
        # conv1_sp = slim.conv2d(self.spatial[:, :, :, 0:ends], 64, [5, 5], padding='VALID', scope='conv1_sp')
        # pool1_sp = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
        # conv2_sp = slim.conv2d(pool1_sp, 32, [5, 5], padding='VALID', scope='conv2_sp')
        # pool2_sp = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
        # pool2_flat_sp = slim.flatten(pool2_sp)

        self.HOI_MLP = nn.Sequential(
            nn.Linear(2048*2, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc, eps=1e-05, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(self.num_fc, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc, eps=1e-05, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5)
        )
        self.HOI_classifier = nn.Sequential(
            nn.Linear(self.num_fc, self.num_classes)

        )
        self.sp_MLP = nn.Sequential(
            nn.Linear(7456, self.num_fc, bias=False), # 5708
            nn.BatchNorm1d(self.num_fc, eps=1e-05, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(self.num_fc, self.num_fc, bias=False),
            nn.BatchNorm1d(self.num_fc, eps=1e-05, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5)
        )
        self.sp_classifier = nn.Sequential(
            nn.Linear(self.num_fc, self.num_classes)
        )

        self.loss = torch.nn.BCEWithLogitsLoss()

    def image_to_head(self, is_training, images):
        # import ipdb
        # ipdb.set_trace()
        # tmp = self.base_model[0](images)[0].tensors
        # tmp = self.base_model[0].normalize(images)
        # self.base_model[0].normalize(images)
        from torchvision.models.detection.image_list import ImageList
        # return self.base_model[1:](tmp)
        return self.base_model(images)

    def sp_to_head(self, spatial):
        # import ipdb
        # ipdb.set_trace()
        return self.Conv_sp(spatial)

    def res5(self, pool5_H, pool5_O, sp, is_training, name):
        return self.h_block(pool5_H), self.o_block(pool5_O)

    def crop_pool_layer(self, bottom, rois, name):
        # import ipdb;ipdb.set_trace()

        from torchvision.ops import roi_pool
        result = roi_pool(
            bottom, rois,
            output_size=(cfg.POOLING_SIZE, cfg.POOLING_SIZE)
        )
        return result


    def res5_ho(self, pool5_HO, is_training, name):
        return self.h_block(pool5_HO)
        # with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        #     if self.model_name.startswith('VCL'):
        #         if self.model_name.__contains__('unique_weights'):
        #             print("unique_weights")
        #             st = -3
        #             reuse = tf.AUTO_REUSE
        #             if name != 'res5':
        #                 reuse = True
        #         else:
        #             st = -2
        #             reuse = tf.AUTO_REUSE
        #         fc7_HO, _ = resnet_v1.resnet_v1(pool5_HO,
        #                                         self.blocks[st:st+1],
        #                                         global_pool=False,
        #                                         include_root_block=False,
        #                                         reuse=reuse,
        #                                         scope=self.scope)
        #     else:
        #         fc7_HO = None
        # return fc7_HO

    def head_to_tail_ho(self, fc7_O, fc7_verbs, fc7_O_raw, fc7_verbs_raw, is_training, name):
        # import ipdb
        # ipdb.set_trace()
        return self.HOI_MLP(torch.cat([fc7_verbs, fc7_O], dim=1))

    def head_to_tail_sp(self, fc7_H, fc7_O, sp, is_training, name):
        # import ipdb
        # ipdb.set_trace()
        return self.sp_MLP(torch.cat([fc7_H, sp], dim=1))

    def region_classification_sp(self, fc7_SHsp, is_training, initializer, name):
        cls_score_sp = self.sp_classifier(fc7_SHsp)
        cls_prob_sp = torch.sigmoid(cls_score_sp)
        self.predictions["cls_score_sp"] = cls_score_sp
        self.predictions["cls_prob_sp"] = cls_prob_sp

        return cls_score_sp


    def region_classification_ho(self, fc7_verbs, is_training, initializer, name, nameprefix = ''):
        cls_score_hoi = self.HOI_classifier(fc7_verbs)
        cls_prob_hoi = torch.sigmoid(cls_score_hoi)
        self.predictions[nameprefix + "cls_score_hoi"] = cls_score_hoi
        self.predictions[nameprefix + "cls_prob_hoi"] = cls_prob_hoi
        if self.model_name.__contains__("VCOCO"):
            # if self.model_name.__contains__('_CL_'):
            #     assert self.num_classes == 222
            #     print(cls_score_hoi, '=============================================')
            if self.model_name.__contains__("VCL_V"):
                self.predictions[nameprefix + "cls_prob_HO"] = cls_prob_hoi if nameprefix == '' else 0
            else:
                self.predictions[nameprefix+"cls_prob_HO"] = self.predictions["cls_prob_sp"] * cls_prob_hoi if nameprefix =='' else 0
        return cls_score_hoi


    def get_compose_boxes(self, h_boxes, o_boxes):
        # import ipdb
        # ipdb.set_trace()
        # h_boxes = torch.unsqueeze(h_boxes, dim=-1)
        # o_boxes = torch.unsqueeze(o_boxes, dim=-1)
        # tmp_box = torch.cat([h_boxes, o_boxes], dim=-1)
        # torch.redu
        # torch.ca
        x1 = torch.where(h_boxes[:, 1] < o_boxes[:, 1], h_boxes[:, 1], o_boxes[:, 1])
        x2 = torch.where(h_boxes[:, 2] < o_boxes[:, 2], h_boxes[:, 2], o_boxes[:, 2])
        y1 = torch.where(h_boxes[:, 3] > o_boxes[:, 3], h_boxes[:, 3], o_boxes[:, 3])
        y2 = torch.where(h_boxes[:, 4] > o_boxes[:, 4], h_boxes[:, 4], o_boxes[:, 4])

        union_boxes = torch.cat([h_boxes[:, 0:1], x1.unsqueeze(-1), x2.unsqueeze(-1),
                                 y1.unsqueeze(-1), y2.unsqueeze(-1)], dim=1)
        return union_boxes


    def forward(self, im_orig, image_id, num_pos, H_boxes, O_boxes, action_HO, Pattern, is_training):
        num_stop = self.get_num_stop(num_pos, H_boxes)
        # ResNet Backbone
        img_w = im_orig.shape[2]
        img_h = im_orig.shape[3]


        head = self.image_to_head(is_training, im_orig)
        sp = self.sp_to_head(Pattern)

        H_boxes = self.convert_rois(H_boxes, head, img_h, img_w)
        O_boxes = self.convert_rois(O_boxes, head, img_h, img_w)

        cboxes = self.get_compose_boxes(H_boxes[:num_stop] if self.model_name.__contains__('VCOCO') else H_boxes, O_boxes)
        pool5_O = self.crop_pool_layer(head, O_boxes, 'Crop_O')
        # import ipdb
        # ipdb.set_trace()
        pool5_H = self.crop_pool_layer(head, H_boxes, 'Crop_H')
        cboxes = cboxes[:num_stop]

        pool5_HO = self.extract_pool5_HO(head, cboxes, H_boxes[:num_stop], is_training, pool5_O, None, name='ho_')
        # print('pool5_O:', pool5_O.shape)
        # further resnet feature
        fc7_H_raw, fc7_O_raw = self.res5(pool5_H, pool5_O, None, is_training, 'res5')
        # print('fc7_H_raw', fc7_H_raw.shape)
        # should be 7x7
        fc7_H = torch.mean(fc7_H_raw, dim=[2, 3])
        fc7_O = torch.mean(fc7_O_raw, dim=[2, 3])
        # print(fc7_O.mean(), fc7_H.mean())
        # import ipdb
        # ipdb.set_trace()
        fc7_H_pos = fc7_H[:num_stop]
        fc7_O_pos = fc7_O[:num_stop]
        fc7_HO_raw = self.res5_ho(pool5_HO, is_training, 'res5')

        fc7_HO = None if fc7_HO_raw is None else torch.mean(fc7_HO_raw, dim=[2, 3])
        # print('fc7_HO', fc7_HO.shape)
        # if not is_training:
        #     # add visualization for test
        #     self.add_visual_for_test(fc7_HO_raw, fc7_H_raw, fc7_O_raw, head, is_training, pool5_O)

        fc7_verbs_raw = fc7_HO_raw
        fc7_verbs = fc7_HO

        # self.score_summaries.update({'orth_HO': fc7_HO,
        #                              'orth_H': fc7_H, 'orth_O': fc7_O})

        fc7_SHsp = self.head_to_tail_sp(fc7_H, fc7_O, sp, is_training, 'fc_HO')
        cls_score_sp = self.region_classification_sp(fc7_SHsp, is_training, None, 'classification')

        # print('verbs')
        if self.model_name.__contains__('VCL_'):
            if not is_training:
                self.test_visualize['fc7_O_feats'] = fc7_O
                self.test_visualize['fc7_verbs_feats'] = fc7_verbs
                self.test_visualize['fc7_H_feats'] = fc7_H_pos

            self.intermediate['fc7_O'] = fc7_O[:num_stop]
            self.intermediate['fc7_verbs'] = fc7_verbs[:num_stop]

            fc7_vo = self.head_to_tail_ho(fc7_O[:num_stop], fc7_verbs[:num_stop], fc7_O_raw, fc7_verbs_raw, is_training, 'fc_HO')
            # print(fc7_vo.mean(), fc7_vo.std())
            cls_score_hoi = self.region_classification_ho(fc7_vo, is_training, None, 'classification')
        else:
            cls_score_hoi = None

        self.score_summaries.update(self.predictions)
        #
        # # label_HO = gt_class_HO_for_verbs
        # label_HO = action_HO[:num_stop]
        # label_sp = action_HO
        #
        # sp_cross_entropy = self.loss(cls_score_sp, label_sp)
        # hoi_cross_entropy = self.loss(cls_score_hoi, label_HO)
        #
        # return sp_cross_entropy + hoi_cross_entropy

    def convert_rois(self, H_boxes, head, img_h, img_w):
        scale_w = head.shape[2] / img_w
        scale_h = head.shape[3] / img_h
        H_boxes[:, 1] = H_boxes[:, 1] * scale_w
        H_boxes[:, 2] = H_boxes[:, 2] * scale_w
        H_boxes[:, 3] = H_boxes[:, 3] * scale_h
        H_boxes[:, 4] = H_boxes[:, 4] * scale_h
        return H_boxes
    # def add_visual_for_test(self, fc7_HO_raw, fc7_H_raw, fc7_O_raw, head, is_training, pool5_O):
    #     self.test_visualize['fc7_H_raw'] = tf.expand_dims(tf.reduce_mean(fc7_H_raw, axis=-1), axis=-1)
    #     self.test_visualize['fc7_O_raw'] = tf.expand_dims(tf.reduce_mean(fc7_O_raw, axis=-1), axis=-1)
    #     if fc7_HO_raw is not None:
    #         self.test_visualize['fc7_HO_raw'] = tf.expand_dims(tf.reduce_mean(fc7_HO_raw, axis=-1), axis=-1)
    #     self.test_visualize['fc7_H_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(fc7_H_raw, 0), tf.float32))
    #     self.test_visualize['fc7_O_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(fc7_O_raw, 0), tf.float32))
    #     if fc7_HO_raw is not None:
    #         self.test_visualize['fc7_HO_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(fc7_HO_raw, 0), tf.float32))
    #     res5_ho_h = self.res5_ho(self.extract_pool5_HO(head, self.H_boxes, is_training, pool5_O, None), is_training,
    #                              'h')
    #     if self.model_name.__contains__('VCL_humans'):
    #         res5_ho_o = self.crop_pool_layer(head, self.O_boxes, 'Crop_HO_h')
    #     else:
    #         res5_ho_o = self.res5_ho(self.extract_pool5_HO(head, self.O_boxes, is_training, pool5_O, None), is_training,
    #                                  'o')
    #     print("res5_ho_o", res5_ho_o, res5_ho_h)
    #     if res5_ho_h is not None and res5_ho_o is not None:
    #         self.test_visualize['res5_ho_H'] = tf.expand_dims(tf.reduce_mean(res5_ho_h, axis=-1), axis=-1)
    #         self.test_visualize['res5_ho_O'] = tf.expand_dims(tf.reduce_mean(res5_ho_o, axis=-1), axis=-1)
    #         self.test_visualize['res5_ho_H_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(res5_ho_h, 0), tf.float32))
    #         self.test_visualize['res5_ho_O_acts_num'] = tf.reduce_sum(tf.cast(tf.greater(res5_ho_o, 0), tf.float32))
    #
    # def add_pose(self, name):
    #     with tf.variable_scope(name) as scope:
    #         conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:][:self.get_num_stop()], 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_pose_map')
    #         pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
    #         conv2_pose_map = slim.conv2d(pool1_pose_map, 16, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_pose_map')
    #         pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
    #         pool2_flat_pose_map = slim.flatten(pool2_pose_map)
    #     return pool2_flat_pose_map
    #
    # def add_pose1(self, name):
    #     with tf.variable_scope(name) as scope:
    #         conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:][:self.get_num_stop()], 64, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_pose_map')
    #         pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
    #         conv2_pose_map = slim.conv2d(pool1_pose_map, 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_pose_map')
    #         pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
    #         pool2_flat_pose_map = slim.flatten(pool2_pose_map)
    #     return pool2_flat_pose_map
    #
    # def add_pose_pattern(self, name = "pose_sp"):
    #     with tf.variable_scope(name) as scope:
    #         conv1_pose_map = slim.conv2d(self.spatial[:self.get_num_stop()], 64, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_sp_pose_map')
    #         pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_sp_pose_map')
    #         conv2_pose_map = slim.conv2d(pool1_pose_map, 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_sp_pose_map')
    #         pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_sp_pose_map')
    #         pool2_flat_pose_map = slim.flatten(pool2_pose_map)
    #     return pool2_flat_pose_map
    #
    # def add_pattern(self, name = 'pattern'):
    #     with tf.variable_scope(name) as scope:
    #         with tf.variable_scope(self.scope, self.scope):
    #             conv1_sp = slim.conv2d(self.spatial[:, :, :, 0:2][:self.get_num_stop()], 64, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv1_sp')
    #             pool1_sp = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
    #             conv2_sp = slim.conv2d(pool1_sp, 32, [5, 5], reuse=tf.AUTO_REUSE, padding='VALID', scope='conv2_sp')
    #             pool2_sp = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
    #             pool2_flat_sp = slim.flatten(pool2_sp)
    #     return pool2_flat_sp

    def get_num_stop(self, H_num, H_boxes):
        """
        following iCAN, spatial pattern include all negative samples. verb-object branch is for positive samples
        self.H_num is the partition for positive sample and negative samples.
        :return:
        """
        num_stop = len(H_boxes)  # for selecting the positive items
        if self.model_name.__contains__('_new') \
                or not self.model_name.startswith('VCL_'):
            print('new Add H_num constrains')
            num_stop = H_num
        elif self.model_name.__contains__('_x5new'):  # contain some negative items
            # I use this strategy cause I found by accident that including
            # some negative samples in the positive samples can improve the performance a bit (abount 0.2%).
            # TODO I think it might have a better solution.
            #  No-Frills Human-Object Interaction Detection provides some support
            #  I think VCL do not depend on this. If someone finds This has important impact on result,
            #  feel happy to contact me.
            H_num_tmp = H_num
            num_stop = num_stop
            num_stop = H_num_tmp + (num_stop - H_num_tmp) // 8
        else:
            num_stop = H_num
        return num_stop

    def get_compose_num_stop(self, H_num, H_boxes):
        num_stop = self.get_num_stop(H_num, H_boxes)
        return num_stop

    def extract_pool5_HO(self, head, cboxes, H_boxes, is_training, pool5_O, head_mask = None, name=''):
        if self.model_name.__contains__('_union'):
            pool5_HO = self.crop_pool_layer(head, cboxes, name + 'Crop_HO')
        elif self.model_name.__contains__('_humans'):
            # print("humans")
            pool5_HO = self.crop_pool_layer(head, H_boxes, name +  'Crop_HO_h')
        else:
            # pool5_HO = self.crop_pool_layer(head, cboxes, 'Crop_HO')
            pool5_HO = None
        return pool5_HO

    def add_loss(self, gt_class_HO, num_stop, device):
        import math
        self.HO_weight = self.HO_weight.to(device)
        if self.model_name.__contains__('_VCOCO'):
            label_H = self.gt_class_H
            label_HO = gt_class_HO
            label_sp = self.gt_class_sp
        else:
            label_H = gt_class_HO[:num_stop]
            # label_HO = gt_class_HO_for_verbs
            label_HO = gt_class_HO[:num_stop]
            label_sp = gt_class_HO
        # if "cls_score_H" in self.predictions:
        #     cls_score_H = self.predictions["cls_score_H"]
        #     """
        #     The re-weighting strategy has an important effect on the performance.
        #     This will also improve largely our baseline in both common and zero-shot setting.
        #     We copy from TIN.
        #     """
        #     if self.model_name.__contains__('_rew'):
        #         cls_score_H = torch.mul(cls_score_H, self.HO_weight)
        #
        #     # H_cross_entropy = torch.enttf.reduce_mean(
        #     #     tf.nn.sigmoid_cross_entropy_with_logits(labels=label_H,
        #     #                                             logits=cls_score_H[:num_stop,  :]))
        #
        # if "cls_score_O" in self.predictions:
        #     cls_score_O = self.predictions["cls_score_O"]
        #     if self.model_name.__contains__('_rew'):
        #         cls_score_O = tf.multiply(cls_score_O, self.HO_weight)
        #     O_cross_entropy = tf.reduce_mean(
        #         tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO,
        #                                                 logits=cls_score_O[:num_stop,  :]))
        #     self.losses['O_cross_entropy'] = O_cross_entropy
        if "cls_score_sp" in self.predictions:
            cls_score_sp = self.predictions["cls_score_sp"]
            if self.model_name.__contains__('_rew'):
                # print('reweight')
                cls_score_sp = torch.mul(cls_score_sp, self.HO_weight)
            # sp_cross_entropy = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(labels=label_sp, logits=cls_score_sp))
            sp_cross_entropy = self.loss(cls_score_sp, label_sp)
            # self.losses['H_cross_entropy'] = H_cross_entropy
            self.losses['sp_cross_entropy'] = sp_cross_entropy

        if self.model_name.startswith('VCL_V_'):
            cls_score_hoi = self.predictions["cls_score_hoi"]
            if self.model_name.__contains__('_rew'):
                # print('reweight')
                cls_score_hoi = torch.mul(cls_score_hoi, self.HO_weight)
            hoi_cross_entropy = self.loss(cls_score_hoi[:num_stop, :], label_HO[:num_stop, :])
            self.losses['hoi_cross_entropy'] = hoi_cross_entropy

            loss = hoi_cross_entropy
        elif self.model_name.startswith('VCL_'):

            tmp_label_HO = gt_class_HO[:num_stop]
            cls_score_hoi = self.predictions["cls_score_hoi"][:num_stop, :]
            if self.model_name.__contains__('_rew'):
                cls_score_hoi = torch.mul(cls_score_hoi, self.HO_weight)

            # tmp_hoi_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tmp_label_HO, logits=cls_score_hoi)

            hoi_cross_entropy = self.loss(cls_score_hoi, tmp_label_HO)

            self.losses['hoi_cross_entropy'] = hoi_cross_entropy

            lamb = 1
            if self.model_name.__contains__('_l05_'):
                lamb = 0.5
            elif self.model_name.__contains__('_l2_'):
                lamb = 2
            elif self.model_name.__contains__('_l0_'):
                lamb = 0
            elif self.model_name.__contains__('_l1_'):
                lamb = 1
            elif self.model_name.__contains__('_l15_'):
                lamb = 1.5
            elif self.model_name.__contains__('_l25_'):
                lamb = 2.5
            elif self.model_name.__contains__('_l3_'):
                lamb = 3
            elif self.model_name.__contains__('_l4_'):
                lamb = 4
            if "cls_score_sp" not in self.predictions:
                sp_cross_entropy = 0
                self.losses['sp_cross_entropy'] = 0
            loss = sp_cross_entropy + hoi_cross_entropy * lamb

        # else:
        #     loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy

        self.losses['total_loss'] = loss
        # self.event_summaries.update(self.losses)
        # print(self.losses)
        # print(self.predictions)
        return loss