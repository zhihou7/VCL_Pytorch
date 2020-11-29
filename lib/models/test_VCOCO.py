# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from Chen Gao, Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp, Get_next_sp_with_pose
from ult.apply_prior import apply_prior

import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'v-coco/coco/images/val2014/COCO_val2014_' + (str(image_id)).zfill(12) + '.jpg'
    # print(im_file)
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection):

    im_orig, im_shape = get_blob(image_id)
    
    blobs = {}
    blobs['H_num']       = 1
   
    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human
            
            # Predict actrion using human appearance only
            blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
            # prediction_H  = net.test_image_H(sess, im_orig, blobs)

            # save image information
            dic = {}
            dic['image_id']   = image_id
            dic['person_box'] = Human_out[2]

            # Predict actrion using human and object appearance 
            Score_obj     = np.empty((0, 4 + 29), dtype=np.float32) 

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object

                    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                    blobs['sp']      = Get_next_sp(Human_out[2], Object[2]).reshape(1, 64, 64, 2)
                    prediction_HO  = net.test_image_HO(sess, im_orig, blobs)

                    if prior_flag == 1:
                        prediction_HO  = apply_prior(Object, prediction_HO)
                    if prior_flag == 2:
                        prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29)
                    if prior_flag == 3:
                        prediction_HO  = apply_prior(Object, prediction_HO)
                        prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29)

                    This_Score_obj = np.concatenate((Object[2].reshape(1,4), prediction_HO[0] * np.max(Object[5])), axis=1)
                    Score_obj      = np.concatenate((Score_obj, This_Score_obj), axis=0)

            # print(prediction_HO.shape, blobs['H_boxes'].shape, blobs['O_boxes'].shape)
            # exit()
            # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
            if Score_obj.shape[0] == 0:
                continue

            # Find out the object box associated with highest action score
            max_idx = np.argmax(Score_obj,0)[4:]

            # agent mAP
            for i in range(29):
                #'''
                # walk, smile, run, stand
                # if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                #     agent_name      = Action_dic_inv[i] + '_agent'
                #     dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                #     continue
                
                # cut
                if i == 2:
                    agent_name = 'cut_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[2]][4 + 2], Score_obj[max_idx[4]][4 + 4])
                    continue 
                if i == 4:
                    continue   

                # eat
                if i == 9:
                    agent_name = 'eat_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[9]][4 + 9], Score_obj[max_idx[16]][4 + 16])
                    continue  
                if i == 16:
                    continue

                # hit
                if i == 19:
                    agent_name = 'hit_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[19]][4 + 19], Score_obj[max_idx[20]][4 + 20])
                    continue  
                if i == 20:
                    continue  

                # These 2 classes need to save manually because there is '_' in action name
                if i == 6:
                    agent_name = 'talk_on_phone_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                    continue 

                # all the rest
                agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
                dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                #'''

                '''
                if i == 6:
                    agent_name = 'talk_on_phone_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                    continue 

                agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
                dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                '''
                   
            # role mAP
            for i in range(29):
                # walk, smile, run, stand. Won't contribute to role mAP
                # if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                #     dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * prediction_H[0][0][i])
                #     continue

                # Impossible to perform this action
                if np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i] == 0:
                   dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

                # Action with >0 score
                else:
                   dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4], np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

            detection.append(dic)


def test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_dir, object_thres, human_thres, prior_flag):


    np.random.seed(cfg.RNG_SEED)
    detection = []
    count = 0

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}


    for line in open(cfg.DATA_DIR + '/' + '/v-coco/data/splits/vcoco_test.ids', 'r'):

        _t['im_detect'].tic()
 
        image_id   = int(line.rstrip())
        
        im_detect(sess, net, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection)

        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 4946, _t['im_detect'].average_time))
        count += 1

    pickle.dump( detection, open( output_dir, "wb" ) )


