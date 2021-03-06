import numpy as np
import tensorflow as tf
from tqdm import tqdm
import zmq
import logging
from tensorflow.python.client import timeline
from os.path import join
from data_utils.nms import nms
from data_utils.normalization import convert_threejs_bbox_with_prob, convert_threejs_coors
from point_viz.converter import PointvizConverter
Converter = PointvizConverter(home='/home/akk/threejs/MTR')

import models.rcnn_config as config
from models import rcnn_model as model
from data_utils.data_generator import DataGenerator

SaiKungDataGenerator = DataGenerator(range_x=config.range_x,
                                     range_y=config.range_y,
                                     range_z=config.range_z)

model_path = '/home/akk/MTR-deploy/checkpoint/model'

input_coors_p, input_features_p, input_num_list_p, _ = model.stage1_inputs_placeholder()
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')


coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list = \
    model.stage1_model(input_coors=input_coors_p,
                       input_features=input_features_p,
                       input_num_list=input_num_list_p,
                       is_training=is_training_p,
                       is_eval=True,
                       trainable=False,
                       mem_saving=False,
                       bn=1.)

bbox_attrs, bbox_conf_logits, bbox_cls_logits, bbox_num_list, bbox_idx = \
    model.stage2_model(coors=coors,
                       features=features,
                       num_list=num_list,
                       roi_attrs=roi_attrs,
                       roi_conf_logits=roi_conf_logits,
                       roi_num_list=roi_num_list,
                       is_training=is_training_p,
                       trainable=False,
                       is_eval=True,
                       mem_saving=False,
                       bn=1.)
bbox_conf = tf.nn.sigmoid(bbox_conf_logits)
bbox_cls = tf.argmax(tf.nn.softmax(bbox_cls_logits, axis=-1), axis=-1)

saver = tf.train.Saver()
tf_config = tf.ConfigProto()
tf_config.gpu_options.visible_device_list = "0"
tf_config.gpu_options.allow_growth = False
tf_config.allow_soft_placement = False
tf_config.log_device_placement = False

context = zmq.Context()
socket = context.socket(zmq.PUSH)
# socket.setsockopt(zmq.SNDHWM, 10)
socket.connect("tcp://127.0.0.1:5555")
logging.info("Pushing zmq binary to tcp://127.0.0.1:5555")

# data = np.array([1., 2.], dtype=np.float32)
# socket.send(data.tobytes("C"))

if __name__ == '__main__':
    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, model_path)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        while True:
            info_tag, batch_input_coors, batch_input_features, batch_input_num_list = \
                next(SaiKungDataGenerator.read_from_zmq())
            output_attrs, output_conf, output_cls, output_coors = \
                sess.run([bbox_attrs, bbox_conf, bbox_cls, roi_coors],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_features,
                                    input_num_list_p: batch_input_num_list,
                                    is_training_p: False},
                         options=run_options,
                         run_metadata=run_metadata)


            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format(show_memory=True)
            with open('timeline.json', 'w') as f:
             f.write(ctf)


            # input_rgbs = np.zeros_like(batch_input_coors) + [255, 255, 255]
            # output_rgbs = np.zeros_like(output_coors) + [255, 0, 0]
            # plot_coors = np.concatenate([batch_input_coors, output_coors], axis=0)
            # plot_rgbs = np.concatenate([input_rgbs, output_rgbs], axis=0)

            # mask = output_conf > 0.3
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
            # pred_bboxes, bbox_collection = nms(pred_bboxes, thres=0.1)
            # # print(len(pred_bboxes))
            # #
            # bbox_count = len(pred_bboxes)
            # frame_id, _, lidar_timestamp, unix_timestamp = info_tag
            # frame_id_b = frame_id.tobytes("C")
            # bbox_count_b = np.array(bbox_count, dtype=np.int32).tobytes("C")
            # lidar_timestamp_b = lidar_timestamp.tobytes("C")
            # unix_timestamp_b = unix_timestamp.tobytes("C")
            # zmq_push_byte = frame_id_b + bbox_count_b + lidar_timestamp_b + unix_timestamp_b
            # if bbox_count > 0:
            #     pred_bboxes = np.array(pred_bboxes, dtype=np.float32)
            #     zmq_push_byte += pred_bboxes.tobytes("C")
            # socket.send(zmq_push_byte)







            # frame_id = deepcopy(np.frombuffer(message[:4], dtype=np.int32))[0]
            # point_count = deepcopy(np.frombuffer(message[4:8], dtype=np.int32))[0]
            # lidar_timestamp = deepcopy(np.frombuffer(message[8:16], dtype=np.float64))[0]
            # unix_timestamp = deepcopy(np.frombuffer(message[16:24], dtype=np.float64))[0]


            # pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes) if len(
            #     pred_bboxes) > 0 else []
            # task_name = "ID_%06d" % (frame_id)

            # Converter.compile(task_name=task_name,
            #                   coors=convert_threejs_coors(plot_coors),
            #                   default_rgb=plot_rgbs,
            #                   bbox_params=pred_bbox_params)


