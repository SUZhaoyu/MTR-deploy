import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
from os.path import join
from data_utils.nms import nms
from copy import deepcopy

from data_utils.normalization import convert_threejs_bbox_with_prob, convert_threejs_coors
from point_viz.converter import PointvizConverter

Converter = PointvizConverter(home='/home/aaeon/threejs/MTR')

from models.configs import roi_config as config
from models import rcnn as model

# from data_utils.data_generator import DataGenerator

# model_path = '/home/aaeon/MTR-deploy/checkpoints/model'

input_coors_p, input_features_p, input_num_list_p, _ = model.inputs_placeholder()
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

base_coors, roi_attrs, num_list, roi_conf_logits, roi_cls_logits = \
    model.model(input_coors=input_coors_p,
                input_features=input_features_p,
                input_num_list=input_num_list_p,
                is_training=is_training_p,
                config=config,
                bn=1.)
roi_conf = tf.nn.sigmoid(roi_conf_logits)
roi_cls = tf.argmax(tf.nn.softmax(roi_cls_logits, axis=-1), axis=-1)
saver = tf.train.Saver()
tf_config = tf.ConfigProto()
tf_config.gpu_options.visible_device_list = "0"
tf_config.gpu_options.allow_growth = False
tf_config.allow_soft_placement = False
tf_config.log_device_placement = False

# context = zmq.Context()
# socket = context.socket(zmq.PUSH)
# socket.bind("tcp://localhost:5558")
# logging.info("Pushing zmq binary to tcp://localhost:5558")
#
# socket.send()

input_coors = np.load("test_data/coors.npy", allow_pickle=True)
input_intensity = np.load("test_data/intensity.npy", allow_pickle=True)
input_num_list = np.load("test_data/num_list.npy", allow_pickle=True)


def input_duplicate(input_coors, input_intensity, input_num_list, ratio=3):
    output_coors = deepcopy(input_coors)
    for i in range(ratio - 1):
        coor_noise_1 = np.random.randn(3) * 0.3
        output_coors = np.concatenate([output_coors, input_coors + coor_noise_1], axis=0)
    output_intensity = np.concatenate([input_intensity] * ratio, axis=0)
    output_num_list = [input_num_list[0] * ratio]

    return output_coors, output_intensity, output_num_list


if __name__ == '__main__':
    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, "checkpoints/model")
        for _ in tqdm(range(1000000)):
            frame_id = np.random.randint(len(input_coors))
            batch_input_coors, batch_input_intensity, batch_input_num_list = \
                input_duplicate(input_coors[frame_id], input_intensity[frame_id], input_num_list[frame_id])
            output_attrs, output_conf, output_cls, output_coors = \
                sess.run([roi_attrs, roi_conf, roi_cls, base_coors],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_intensity,
                                    input_num_list_p: batch_input_num_list,
                                    is_training_p: False})

            # if frame_id % 100 == 0:
            # input_rgbs = np.zeros_like(batch_input_coors) + [255, 255, 255]
            # output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]
            # plot_coors = np.concatenate([batch_input_coors, output_coors], axis=0)
            # plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)

            # mask = output_conf > 0.7
            # output_conf = output_conf[mask]
            # output_cls = output_cls[mask]
            # output_bboxes = output_attrs[mask, :]
            # w = output_bboxes[:, 0]
            # l = output_bboxes[:, 1]
            # h = output_bboxes[:, 2]
            # x = output_bboxes[:, 3]
            # y = output_bboxes[:, 4]
            # z = output_bboxes[:, 5]
            # r = output_bboxes[:, 6]
            # c = output_cls
            # pred_bboxes = np.stack([w, l, h, x, y, z, r, c], axis=-1)
            # pred_bboxes = np.concatenate([pred_bboxes, np.expand_dims(output_conf, axis=-1)], axis=-1)
            # pred_bboxes, bbox_collection = nms(pred_bboxes, thres=1e-3)

            # pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes) if len(
            #     pred_bboxes) > 0 else []
            # task_name = "ID_%06d" % (frame_id)

            # Converter.compile(task_name=task_name,
            #                   coors=convert_threejs_coors(plot_coors),
            #                   default_rgb=plot_rgbs,
            #                   bbox_params=pred_bbox_params)


