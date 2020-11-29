# --------------------------------------------------------
# Tensorflow VCL
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhi Hou, based on code from iCAN and TIN
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.tools import get_convert_matrix
from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp, Get_next_sp_with_pose

import cv2
import pickle
import numpy as np
import glob



# This strategy is based on TIN and is effective
human_num_thres = 4
object_num_thres = 4

def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def test_net_data_api1(sess, net, output_dir, h_box, o_box, o_cls, h_score, o_score, image_id):
    detection = {}
    verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix()
    # prediction_HO  = net.test_image_HO(sess, im_orig, blobs)
    # timers
    ones = np.ones([1, 600], np.float32)
    _t = {'im_detect': Timer(), 'misc': Timer()}
    last_img_id = -1
    count = 0

    fuse_res = tf.constant(0)

    obj_scores = tf.constant(0)
    objid = tf.constant(0)
    obj_scores = tf.constant(0)
    _t['im_detect'].tic()
    while True:
        _t['im_detect'].tic()

        from tensorflow.python.framework.errors_impl import InvalidArgumentError
        try:
            pH, pO, pSp, pVerbs, pSpHO, pFuse, f_obj_score, f_obj_cls, _h_box, _o_box, _o_cls, _h_score, _o_score, _image_id = sess.run([
                net.predictions["cls_prob_H"] if 'cls_prob_H' in net.predictions else h_box,
                net.predictions["cls_prob_O"] if 'cls_prob_O' in net.predictions else h_box,
                net.predictions["cls_prob_sp"] if 'cls_prob_sp' in net.predictions else h_box,
                net.predictions["cls_prob_hoi"] if 'cls_prob_hoi' in net.predictions else h_box,
                net.predictions["cls_prob_spverbs"] if 'cls_prob_spverbs' in net.predictions else h_box,
                fuse_res if 'cls_prob_sp' in net.predictions else h_box, obj_scores, objid,
                                        h_box, o_box, o_cls, h_score, o_score, image_id])
            # print(pFuse.shape, f_obj_score.shape, f_obj_cls.shape)
        except InvalidArgumentError as e:
            # cls_prob_HO = np.zeros(shape=[blobs['sp'].shape[0], self.num_classes])
            raise e
        except tf.errors.OutOfRangeError:
            print('END')
            break

        # if net.model_name.__contains__('cosine'):
        #     temp = [[_h_box[i], _o_box[i], _o_cls[i], 0, _h_score[i], _o_score[i], pH[i], pO[i], pSp[i], pVerbs[i], pSpHO[i], pFuse[i], f_obj_score[i], f_obj_cls[i]] for i in range(len(_h_box))]
        # else:
        temp = [[_h_box[i], _o_box[i], _o_cls[i], 0, _h_score[i], _o_score[i], pH[i], pO[i], pSp[i], pVerbs[i], pSpHO[i]] for i in range(len(_h_box))]

        # detection[_image_id] = temp
        if _image_id in detection:
            detection[_image_id].extend(temp)
        else:
            detection[_image_id] = temp

        _t['im_detect'].toc()
        count += 1

        print('\rmodel: {} im_detect: {:d}/{:d}  {:d}, {:.3f}s'.format(net.model_name, count, 10566, _image_id, _t['im_detect'].average_time), end='', flush=True)

    # TODO remove
    # pickle.dump(detection, open('test_new.pkl', 'wb'))
    pickle.dump(detection, open(output_dir, "wb"))
    del detection
    import gc
    gc.collect()


def obtain_test_dataset(object_thres, human_thres, dataset_name='test2015'):
    from sys import version_info
    if dataset_name == 'test2015':
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"),
                                    encoding='latin1')
            Test_RCNN = obtain_obj_boxes('default')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"))
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
            i += 1
            # if i > 30: # TODO remove
            #     break
            image_id = int(line[-9:-4])
            # save image information
            im_orig, im_shape = get_blob(image_id)
            mask_all = np.zeros(shape=(1, im_shape[0], im_shape[1], 1), dtype=np.float32)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            for Human_out in Test_RCNN[image_id]:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)

                    for Object in Test_RCNN[image_id]:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append(np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            obj_box = np.array(
                                [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                            blobs['O_boxes'].append(obj_box)
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))

                            mask = np.zeros(shape=(im_shape[0], im_shape[1], 1), dtype=np.float32)
                            obj_box = obj_box[1:].astype(np.int32)
                            # print(obj_box)
                            # print(obj_box, blobs['O_boxes'])
                            mask[obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                            mask_all[:, obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1
                            # from skimage import transform
                            # mask = transform.resize(mask, [im_shape[0] // 16, im_shape[1] // 16, 1], order=0,
                            #                         preserve_range=True)
                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                # print('None ', image_id)
                continue
            # print(im_orig.shape, image_id, end=' ')
            # for k, v in blobs.items():
            #     if not k == 'H_num':
            #         blobs[k] = np.asarray(v)
            #         print(k, blobs[k].shape, end=' ')
            # print('\n')
            im_mask = np.multiply(im_orig, mask_all)
            im_orig = np.concatenate([im_orig, im_mask], axis=0)
            start = 0
            stride = 200
            while start < blobs['H_num']: # save GPU memory
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]

                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride
                # print('b_temp' , im_orig.shape, image_id, end=' ')
                # for k, v in blobs.items():
                #     if not k == 'H_num':
                #         blobs[k] = np.asarray(v)
                #         print(k, blobs[k].shape, end=' ')
                # print('\n')

                yield im_orig, b_temp, image_id
            # yield im_orig, blobs, image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32}, tf.int32,),
                                             output_shapes=(tf.TensorShape([2, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None])},
                                                            tf.TensorShape([]))
                                             )
    dataset = dataset.prefetch(100)
    # dataset = dataset.repeat(100000) # TODO improve
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id


def obtain_test_dataset1(object_thres, human_thres, dataset_name='test2015', test_type='default',
                         has_human_threhold=True, stride = 200, model_name=''):
    print('================================================================================')
    print(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg', glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'))
    from sys import version_info
    if dataset_name == 'test2015':
        print(test_type, version_info.major)
        if version_info.major == 3:
            # Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose
            Test_RCNN = obtain_obj_boxes(test_type)
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb"))
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        # for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
        for image_id in Test_RCNN:
            i += 1
            # if i > 30: # TODO remove
            #     break
            # image_id = int(line[-9:-4])
            # save image information
            im_orig, im_shape = get_blob(image_id)
            mask_all = np.zeros(shape=(1, im_shape[0], im_shape[1], 1), dtype=np.float32)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            for Human_out in Test_RCNN[image_id]:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)

                    for Object in Test_RCNN[image_id]:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]])
                            obj_box = [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]
                            blobs['O_boxes'].append(obj_box)
                            # print(len(Human_out), len(Human_out[6]))
                            # if len(Human_out) <= 6 or (len(Human_out) == 7 and (Human_out[6] is None or len(Human_out[6]) == 80)):
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                                # blobs['sp'][-1] = np.concatenate([blobs['sp'][-1], blobs['sp'][-1][:,:,:1]], axis=-1)
                                # print('humanbox')
                            # else:
                            #     blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]))

                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])

            """
            Notice: This strategy is based on TIN and is effective! This could improve the performance around 1.%
            Re-weighting and This strategy improve our baseline to 18.04%.
            """
            if blobs['H_num'] == 0 and has_human_threhold:
                print('\rDealing with zero-sample test Image ' + str(image_id), end='', flush=True)

                list_human_included = []
                list_object_included = []
                Human_out_list = []
                Object_list = []

                test_pair_all = Test_RCNN[image_id]
                length = len(test_pair_all)

                flag_continue_searching = 1

                while (len(list_human_included) < human_num_thres) or (
                        len(list_object_included) < object_num_thres):
                    h_max = [-1, -1.0]
                    o_max = [-1, -1.0]
                    flag_continue_searching = 0
                    for i in range(length):
                        if test_pair_all[i][1] == 'Human':
                            if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(
                                    list_human_included) < human_num_thres:
                                h_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1
                        else:
                            if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(
                                    list_object_included) < object_num_thres:
                                o_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1

                    if flag_continue_searching == 0:
                        break

                    list_human_included.append(h_max[0])
                    list_object_included.append(o_max[0])

                    Human_out_list.append(test_pair_all[h_max[0]])
                    Object_list.append(test_pair_all[o_max[0]])

                for Human_out in Human_out_list:
                    for Object in Object_list:

                        blobs['H_num'] += 1
                        blobs['H_boxes'].append([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]])
                        obj_box = [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]
                        blobs['O_boxes'].append(obj_box)
                        # if len(Human_out) <= 6 or (len(Human_out) == 7 and (Human_out[6] is None)):
                        # if len(Human_out) <= 6:
                        blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                            # blobs['sp'][-1] = np.concatenate([blobs['sp'][-1], blobs['sp'][-1][:,:,:1]], axis=-1)
                        # else:
                        #     blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]))
                        blobs['O_cls'].append(Object[4])
                        blobs['H_score'].append(Human_out[5])
                        blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                # print('None ', image_id)
                continue
            # print(im_orig.shape, image_id, end=' ')
            # for k, v in blobs.items():
            #     if not k == 'H_num':
            #         blobs[k] = np.asarray(v)
            #         print(k, blobs[k].shape, end=' ')
            # print('\n')
            start = 0
            # stride = 200
            while start < blobs['H_num']:
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]


                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride
                # print('b_temp' , im_orig.shape, image_id, end=' ')
                # for k, v in blobs.items():
                #     if not k == 'H_num':
                #         blobs[k] = np.asarray(v)
                #         print(k, blobs[k].shape, end=' ')
                # print('\n')
                # print(b_temp['sp'].shape)
                b_temp['H_boxes'] = np.asarray(b_temp['H_boxes'], np.float32)
                b_temp['O_boxes'] = np.asarray(b_temp['O_boxes'], np.float32)
                b_temp['sp'] = np.asarray(b_temp['sp'], np.float32)
                yield im_orig, b_temp, image_id

    return generator1()

def obtain_obj_boxes(test_type):

    if test_type == 'debug':
        pass
    else:
        # Test_RCNN = pickle.load(open('/opt/data/private/pkl/Test_HICO_res101_rex2_0059999_FPN_hico.pkl', "rb"))
        #
        print('Default ===========')
        Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"),
                                encoding='latin1')
    return Test_RCNN


def obtain_test_dataset2(object_thres, human_thres, dataset_name='test2015', test_type='default'):
    from sys import version_info
    if dataset_name == 'test2015':
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb"),
                                    encoding='latin1')
            # Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose
            if test_type != 'default':
                Test_RCNN = pickle.load(
                    open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_DET_finetune.pkl', "rb"),
                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb"))
    else:
        if version_info.major == 3:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"),
                                    encoding='latin1')
        else:
            Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    def generator1():
        np.random.seed(cfg.RNG_SEED)
        i = 0
        for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/'+dataset_name+'/*.jpg'):
            i += 1
            # if i > 30: # TODO remove
            #     break
            image_id = int(line[-9:-4])
            # save image information
            im_orig, im_shape = get_blob(image_id)
            mask_all = np.zeros(shape=(1, im_shape[0], im_shape[1], 1), dtype=np.float32)
            blobs = {}

            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['O_cls'] = []
            blobs['H_score'] = []
            blobs['O_score'] = []
            for Human_out in Test_RCNN[image_id]:
                if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

                    # blobs['H_boxes'] = np.array(
                    #     [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)

                    for Object in Test_RCNN[image_id]:
                        if (np.max(Object[5]) > object_thres) and not (
                                np.all(Object[2] == Human_out[2])):  # This is a valid object

                            blobs['H_num'] += 1
                            blobs['H_boxes'].append(np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                            obj_box = np.array(
                                [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                            blobs['O_boxes'].append(obj_box)
                            if len(Human_out) <= 6:
                                blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                            else:
                                blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]))


                            blobs['O_cls'].append(Object[4])
                            blobs['H_score'].append(Human_out[5])
                            blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                print('\rDealing with zero-sample test Image ' + str(image_id), end='', flush=True)

                list_human_included = []
                list_object_included = []
                Human_out_list = []
                Object_list = []

                test_pair_all = Test_RCNN[image_id]
                length = len(test_pair_all)

                flag_continue_searching = 1

                while (len(list_human_included) < human_num_thres) or (
                        len(list_object_included) < object_num_thres):
                    h_max = [-1, -1.0]
                    o_max = [-1, -1.0]
                    flag_continue_searching = 0
                    for i in range(length):
                        if test_pair_all[i][1] == 'Human':
                            if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(
                                    list_human_included) < human_num_thres:
                                h_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1
                        else:
                            if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(
                                    list_object_included) < object_num_thres:
                                o_max = [i, np.max(test_pair_all[i][5])]
                                flag_continue_searching = 1

                    if flag_continue_searching == 0:
                        break

                    list_human_included.append(h_max[0])
                    list_object_included.append(o_max[0])

                    Human_out_list.append(test_pair_all[h_max[0]])
                    Object_list.append(test_pair_all[o_max[0]])

                for Human_out in Human_out_list:
                    for Object in Object_list:

                        blobs['H_num'] += 1
                        blobs['H_boxes'].append(
                            np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                        obj_box = np.array(
                            [0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]])
                        blobs['O_boxes'].append(obj_box)
                        if len(Human_out) <= 6:
                            blobs['sp'].append(Get_next_sp(Human_out[2], Object[2]))
                        else:
                            blobs['sp'].append(Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]))


                        blobs['O_cls'].append(Object[4])
                        blobs['H_score'].append(Human_out[5])
                        blobs['O_score'].append(Object[5])

            if blobs['H_num'] == 0:
                # print('None ', image_id)
                continue
            # print(im_orig.shape, image_id, end=' ')
            # for k, v in blobs.items():
            #     if not k == 'H_num':
            #         blobs[k] = np.asarray(v)
            #         print(k, blobs[k].shape, end=' ')
            # print('\n')
            im_mask = np.multiply(im_orig, mask_all)
            im_orig = np.concatenate([im_orig, im_mask], axis=0)
            start = 0
            stride = 200
            while start < blobs['H_num']:
                b_temp = {}
                for k ,v in blobs.items():
                    if not k == 'H_num':
                        b_temp[k] = blobs[k][start:start+stride]

                b_temp['H_num'] = min(start + stride, blobs['H_num']) - start
                start += stride

                yield im_orig, b_temp, image_id

    dataset = tf.data.Dataset.from_generator(generator1, output_types=(
        tf.float32, {'H_num': tf.int32, 'H_boxes': tf.float32, 'O_boxes': tf.float32, 'sp': tf.float32,
                     'O_cls': tf.float32, 'H_score': tf.float32, 'O_score': tf.float32}, tf.int32,),
                                             output_shapes=(tf.TensorShape([2, None, None, 3]),
                                                            {'H_num': tf.TensorShape([]),
                                                             'H_boxes': tf.TensorShape([None, 5]),
                                                             'O_boxes': tf.TensorShape([None, 5]),
                                                             'sp': tf.TensorShape([None, 64, 64, 2]),
                                                             'O_cls': tf.TensorShape([None]),
                                                             'H_score': tf.TensorShape([None]),
                                                             'O_score': tf.TensorShape([None])},
                                                            tf.TensorShape([]))
                                             )
    dataset = dataset.prefetch(100)
    # dataset = dataset.repeat(100000) # TODO improve
    iterator = dataset.make_one_shot_iterator()
    image, blobs, image_id = iterator.get_next()
    return image, blobs, image_id