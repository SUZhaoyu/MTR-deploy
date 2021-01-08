import tensorflow as tf

import models.rcnn_config as config
from models.tf_ops.custom_ops import roi_pooling, get_roi_bbox, roi_filter, get_bbox
from models.utils.iou_utils import cal_3d_iou
from models.utils.loss_utils import get_masked_average, focal_loss
from models.utils.model_layers import point_conv, fully_connected, conv_3d
from models.utils.ops_wrapper import get_roi_attrs, get_bbox_attrs

anchor_size = config.anchor_size
eps = tf.constant(1e-6)

model_params = {'xavier': config.xavier,
                'stddev': config.stddev,
                'activation': config.activation}

def stage1_inputs_placeholder(input_channels=1,
                              bbox_padding=config.aug_config['nbbox']):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage1_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='stage1_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage1_input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 8], name='stage1_input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p


def stage2_inputs_placeholder(input_feature_channels=config.base_params[sorted(config.base_params.keys())[-1]]['c_out'],
                              bbox_padding=config.aug_config['nbbox']):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage2_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_feature_channels], name='stage2_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage2_input_num_list_p')
    input_roi_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage_input_roi_coors_p')
    input_roi_attrs_p = tf.placeholder(tf.float32, shape=[None, 7], name='stage2_input_roi_attrs_p')
    input_roi_conf_p = tf.placeholder(tf.float32, shape=[None], name='stage2_input_roi_conf_p')
    input_roi_num_list_p = tf.placeholder(tf.int32, shape=[None], name='stage2_input_roi_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 8], name='stage2_input_bbox_p')
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
    dimension_params = {'dimension': config.dimension,
                        'offset': config.offset}

    base_params = config.base_params
    coors, features, num_list = input_coors, input_features, input_num_list

    with tf.variable_scope("stage1"):
        # =============================== STAGE-1 [base] ================================
        for layer_name in sorted(base_params.keys()):
            coors, features, num_list = point_conv(input_coors=coors,
                                                   input_features=features,
                                                   input_num_list=num_list,
                                                   layer_params=base_params[layer_name],
                                                   dimension_params=dimension_params,
                                                   scope="stage1_" + layer_name,
                                                   is_training=is_training,
                                                   trainable=trainable,
                                                   mem_saving=mem_saving,
                                                   model_params=model_params,
                                                   bn_decay=bn)

        # =============================== STAGE-1 [rpn] ================================

        roi_coors, roi_features, roi_num_list = point_conv(input_coors=coors,
                                                           input_features=features,
                                                           input_num_list=num_list,
                                                           layer_params=config.rpn_params,
                                                           dimension_params=dimension_params,
                                                           scope="stage1_rpn_conv",
                                                           is_training=is_training,
                                                           trainable=trainable,
                                                           mem_saving=mem_saving,
                                                           model_params=model_params,
                                                           bn_decay=bn)

        roi_logits = fully_connected(input_points=roi_features,
                                     num_output_channels=config.output_attr-config.cls_num,
                                     drop_rate=0.,
                                     model_params=model_params,
                                     scope='stage1_rpn_fc',
                                     is_training=is_training,
                                     trainable=trainable,
                                     last_layer=True)

        roi_attrs = get_roi_attrs(input_logits=roi_logits,
                                  base_coors=roi_coors,
                                  anchor_size=anchor_size,
                                  is_eval=is_eval)

        roi_conf_logits = roi_logits[:, 7]

        return coors, features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list

def stage2_model(coors,
                 features,
                 num_list,
                 roi_attrs,
                 roi_conf_logits,
                 roi_num_list,
                 is_training,
                 trainable,
                 is_eval,
                 mem_saving,
                 bn):
    instance_max = 0 if is_eval else config.max_roi_per_instance
    with tf.variable_scope("stage2"):
        roi_conf = tf.nn.sigmoid(roi_conf_logits)
        bbox_roi_attrs, bbox_num_list, bbox_idx = roi_filter(input_roi_attrs=roi_attrs,
                                                             input_roi_conf=roi_conf,
                                                             input_num_list=roi_num_list,
                                                             conf_thres=config.roi_thres,
                                                             instance_max=instance_max)

        bbox_voxels = roi_pooling(input_coors=coors,
                                  input_features=features,
                                  roi_attrs=bbox_roi_attrs,
                                  input_num_list=num_list,
                                  roi_num_list=bbox_num_list,
                                  voxel_size=config.roi_voxel_size,
                                  pooling_size=5.)

        for i in range(config.roi_voxel_size // 2):
            bbox_voxels = conv_3d(input_voxels=bbox_voxels,
                                  layer_params=config.refine_params,
                                  scope="stage2_refine_conv_{}".format(i),
                                  is_training=is_training,
                                  trainable=trainable,
                                  model_params=model_params,
                                  mem_saving=mem_saving,
                                  bn_decay=bn)

        bbox_features = tf.squeeze(bbox_voxels, axis=[1])

        bbox_logits = fully_connected(input_points=bbox_features,
                                      num_output_channels=config.output_attr,
                                      drop_rate=0.,
                                      model_params=model_params,
                                      scope='stage2_refine_fc',
                                      is_training=is_training,
                                      trainable=trainable,
                                      last_layer=True)

        bbox_attrs = get_bbox_attrs(input_logits=bbox_logits,
                                    input_roi_attrs=bbox_roi_attrs,
                                    is_eval=is_eval)

        bbox_conf_logits = bbox_logits[:, 7]
        bbox_cls_logits = bbox_logits[:, 8:]

    return bbox_attrs, bbox_conf_logits, bbox_cls_logits, bbox_num_list, bbox_idx
