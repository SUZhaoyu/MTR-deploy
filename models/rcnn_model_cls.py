# import horovod.tensorflow as hvd
import tensorflow as tf

import models.rcnn_config as config
from models.tf_ops.loader.bbox_utils import get_roi_bbox, get_bbox
from models.tf_ops.loader.others import roi_filter, rotated_nms3d_idx
from models.tf_ops.loader.pooling import la_roi_pooling_fast
from models.utils.iou_utils import cal_3d_iou
from models.utils.loss_utils import get_masked_average, focal_loss, smooth_l1_loss, get_dir_cls
from models.utils.model_blocks import point_conv, conv_1d, conv_3d, point_conv_res, conv_3d_res, point_conv_concat
from models.utils.layers_wrapper import get_roi_attrs, get_bbox_attrs

anchor_size = config.anchor_size
eps = tf.constant(1e-6)

model_params = {'xavier': config.xavier,
                'stddev': config.stddev,
                'activation': config.activation,
                'padding': 0.,}

def stage1_inputs_placeholder(input_channels=1,
                              bbox_padding=config.aug_config['nbbox']):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage1_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='stage1_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage1_input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 9], name='stage1_input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p

# config.base_params_inference[sorted(config.base_params_inference.keys())[-1]]['c_out']
def stage2_inputs_placeholder(input_feature_channels=385,
                              bbox_padding=config.aug_config['nbbox']):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage2_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_feature_channels], name='stage2_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage2_input_num_list_p')
    input_roi_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage_input_roi_coors_p')
    input_roi_attrs_p = tf.placeholder(tf.float32, shape=[None, 7], name='stage2_input_roi_attrs_p')
    input_roi_conf_p = tf.placeholder(tf.float32, shape=[None], name='stage2_input_roi_conf_p')
    input_roi_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage2_input_roi_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 9], name='stage2_input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_roi_coors_p, \
           input_roi_attrs_p, input_roi_conf_p, input_roi_num_list_p, input_bbox_p


def stage1_model(input_coors,
                 input_features,
                 input_num_list,
                 is_training,
                 trainable,
                 is_eval,
                 mem_saving,
                 bn):
    # if is_eval:
    #     dimension_params = {'dimension': config.dimension_testing,
    #                         'offset': config.offset_testing}
    # else:
    dimension_params = {'dimension': config.dimension_training,
                        'offset': config.offset_training}


    # base_params = config.base_params_inference if not is_eval else config.base_params_inference
    # rpn_params = config.rpn_params_inference if not is_eval else config.rpn_params_inference

    base_params = config.base_params_inference
    # rpn_params = config.rpn_params_inference

    coors, features, num_list = input_coors, input_features, input_num_list
    concat_features = features
    voxel_idx, center_idx = None, None

    with tf.variable_scope("stage1"):
        # =============================== STAGE-1 [base] ================================

        for i, layer_name in enumerate(sorted(base_params.keys())):
            coors, features, num_list, voxel_idx, center_idx, concat_features = \
                point_conv_concat(input_coors=coors,
                                  input_features=features,
                                  concat_features=concat_features,
                                  input_num_list=num_list,
                                  voxel_idx=voxel_idx,
                                  center_idx=center_idx,
                                  layer_params=base_params[layer_name],
                                  dimension_params=dimension_params,
                                  grid_buffer_size=config.grid_buffer_size,
                                  output_pooling_size=config.output_pooling_size,
                                  scope="stage1_" + layer_name,
                                  is_training=is_training,
                                  trainable=trainable,
                                  mem_saving=mem_saving,
                                  model_params=model_params,
                                  bn_decay=bn)

        # =============================== STAGE-1 [rpn] ================================

        roi_features = concat_features
        roi_coors = coors
        roi_num_list = num_list


        cls_features = conv_1d(input_points=concat_features,
                               num_output_channels=256,
                               drop_rate=0.,
                               model_params=model_params,
                               scope='stage1_rpn_cls_fc_0',
                               is_training=is_training,
                               trainable=trainable,
                               second_last_layer=True)

        cls_logits = conv_1d(input_points=cls_features,
                             num_output_channels=3,
                             drop_rate=0.,
                             model_params=model_params,
                             scope='stage1_rpn_cls_fc_1',
                             is_training=is_training,
                             trainable=trainable,
                             last_layer=True)


        roi_features = conv_1d(input_points=concat_features,
                               num_output_channels=256,
                               drop_rate=0.,
                               model_params=model_params,
                               scope='stage1_rpn_fc_0',
                               is_training=is_training,
                               trainable=trainable,
                               second_last_layer=True)

        roi_logits = conv_1d(input_points=roi_features,
                             num_output_channels=config.output_attr,
                             drop_rate=0.,
                             model_params=model_params,
                             scope='stage1_rpn_fc_1',
                             is_training=is_training,
                             trainable=trainable,
                             last_layer=True)

        roi_attrs = get_roi_attrs(input_logits=roi_logits,
                                  base_coors=coors,
                                  anchor_size=anchor_size,
                                  is_eval=is_eval)

        roi_conf_logits = roi_logits[:, 7]


        return coors, concat_features, num_list, roi_coors, roi_attrs, roi_conf_logits, cls_logits, roi_num_list


