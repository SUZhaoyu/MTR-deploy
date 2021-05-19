import numpy as np
import tensorflow as tf
from tqdm import tqdm
import zmq
import logging
from datetime import datetime
import os
import json
import tzlocal
from tensorflow.python.client import timeline
from os.path import join
from data_utils.nms import nms
from models.tf_ops.loader.others import rotated_nms3d_idx
from data_utils.normalization import convert_threejs_bbox_with_prob, convert_threejs_coors
from point_viz.converter import PointvizConverter
from data_utils.json_converter import convert_json

os.system('rm -rf /home/akk/threejs/MTR')
Converter = PointvizConverter(home='/home/akk/threejs/MTR')


import models.rcnn_config as config
from models import rcnn_model_cls as model
from data_utils.data_generator import DataGenerator

SaiKungDataGenerator = DataGenerator(range_x=config.range_x,
                                     range_y=config.range_y,
                                     range_z=config.range_z)

model_path = '/home/akk/MTR-deploy/checkpoint-cls/model_0.6689047517789147'

input_coors_p, input_features_p, input_num_list_p, _ = model.stage1_inputs_placeholder()
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')


coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, cls_logits, roi_num_list = \
    model.stage1_model(input_coors=input_coors_p,
                       input_features=input_features_p,
                       input_num_list=input_num_list_p,
                       is_training=is_training_p,
                       is_eval=True,
                       trainable=False,
                       mem_saving=False,
                       bn=1.)

roi_conf = tf.nn.sigmoid(roi_conf_logits)
nms_idx = rotated_nms3d_idx(roi_attrs, roi_conf, nms_overlap_thresh=0.15, nms_conf_thres=0.5)
roi_coors = tf.gather(roi_coors, nms_idx, axis=0)
roi_attrs = tf.gather(roi_attrs, nms_idx, axis=0)
cls_logits = tf.gather(cls_logits, nms_idx, axis=0)
roi_conf_logits = tf.gather(roi_conf_logits, nms_idx, axis=0)
roi_num_list = tf.expand_dims(tf.shape(nms_idx)[0], axis=0)

roi_conf = tf.nn.sigmoid(roi_conf_logits)
cls_prob = tf.nn.softmax(cls_logits)
pred_cls = tf.argmax(cls_prob, axis=1)


saver = tf.train.Saver()
tf_config = tf.ConfigProto()
tf_config.gpu_options.visible_device_list = "0"
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = False
tf_config.log_device_placement = False

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.setsockopt(zmq.SNDHWM, 10)
socket.connect("tcp://127.0.0.1:5555")
logging.info("Pushing zmq binary to tcp://127.0.0.1:5555")

# context_flag = zmq.Context()
# socket_flag = context_flag.socket(zmq.PUSH)
# socket_flag.setsockopt(zmq.CONFLATE, 1)
# socket_flag.connect("tcp://127.0.0.1:5558")
# logging.info("Pushing zmq binary to tcp://127.0.0.1:5558")

# data = np.array([1., 2.], dtype=np.float32)
# socket.send(data.tobytes("C"))

if __name__ == '__main__':
    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, model_path)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # while True:
        for frame_id in tqdm(range(100000)):
            info_tag, batch_input_coors, batch_input_features, batch_input_num_list = \
                next(SaiKungDataGenerator.read_from_zmq())
            output_attrs, output_conf, output_cls_conf, output_cls, output_coors = \
                sess.run([roi_attrs, roi_conf, cls_prob, pred_cls, roi_coors],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_features,
                                    input_num_list_p: batch_input_num_list,
                                    is_training_p: False})

            output_idx = output_conf > 0.8
            output_bboxes = output_attrs[output_idx]
            output_conf = output_conf[output_idx]
            output_cls_conf = output_cls_conf[output_idx]
            output_cls = output_cls[output_idx]

            # print(np.mean(batch_input_features), np.std(batch_input_features))

            w = output_bboxes[:, 0]
            l = output_bboxes[:, 1]
            h = output_bboxes[:, 2]
            x = output_bboxes[:, 3]
            y = output_bboxes[:, 4]
            z = output_bboxes[:, 5]
            r = output_bboxes[:, 6]
            c = output_cls
            for i in range(len(c)):
                if c[i] > 0 and output_cls_conf[i, c[i]] <= 0.7:
                    c[i] = 0
            pred_bboxes = np.stack([w, l, h, x, y, z, r, c], axis=-1)
            pred_bboxes = np.concatenate([pred_bboxes, np.expand_dims(output_conf, axis=-1), output_cls_conf], axis=-1)

            original_length = len(pred_bboxes)
            if original_length > 0:
                pred_bboxes = pred_bboxes[np.logical_not(np.isnan(np.sum(pred_bboxes, axis=-1))), :]
                if len(pred_bboxes) != original_length:
                    now = datetime.now()
                    current_time = now.strftime("%m/%d/%Y-%H:%M:%S")
                    logging.warning("Warning: NaN detected @ {}".format(current_time))

            original_length = len(pred_bboxes)
            if original_length > 0:
                pred_bboxes = pred_bboxes[np.max(pred_bboxes[:, :3], axis=-1) < 5., :]
                if len(pred_bboxes) != original_length:
                    now = datetime.now()
                    current_time = now.strftime("%m/%d/%Y-%H:%M:%S")
                    logging.warning("Warning: Crazy bbox dimension (>=5 meter) detected @ {}".format(current_time))


            bbox_count = len(pred_bboxes)
            frame_id, _, lidar_timestamp, unix_timestamp = info_tag
            frame_id_b = frame_id.tobytes("C")
            bbox_count_b = np.array(bbox_count, dtype=np.int32).tobytes("C")
            lidar_timestamp_b = lidar_timestamp.tobytes("C")
            unix_timestamp_b = unix_timestamp.tobytes("C")
            zmq_push_byte = frame_id_b + bbox_count_b + lidar_timestamp_b + unix_timestamp_b
            if bbox_count > 0:
                pred_bboxes = np.array(pred_bboxes, dtype=np.float32)
                zmq_push_byte += pred_bboxes.tobytes("C")
            socket.send(zmq_push_byte)





            # if np.sum(output_cls > 0):
            #     socket_flag.send(frame_id_b)
            #     output_json = convert_json(pred_bboxes)
            #     local_timezone = tzlocal.get_localzone()  # get pytz timezone
            #     local_time = datetime.fromtimestamp(unix_timestamp, local_timezone)
            #     # print(output_json)
            #     time_id = local_time.strftime("%Y_%m_%d=%H_%M_%S_%f")
            #     print(time_id[:-3])
            #     with open('/home/akk/lugg_json/{}.bin.json'.format(time_id[:-3]), 'w') as f:
            #         json.dump(output_json, f)





            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format(show_memory=True)
            # with open('timeline.json', 'w') as f:
            #  f.write(ctf)


            # input_rgbs = np.zeros_like(batch_input_coors) + [255, 255, 255]
            # output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]
            # plot_coors = np.concatenate([batch_input_coors, output_coors], axis=0)
            # plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)
            #
            #
            # pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes) if len(
            #     pred_bboxes) > 0 else []
            # task_name = "ID_%06d" % (frame_id)
            # if frame_id % 10 == 0:
            #     Converter.compile(task_name=task_name,
            #                       coors=convert_threejs_coors(plot_coors),
            #                       default_rgb=plot_rgbs,
            #                       bbox_params=pred_bbox_params)
                

