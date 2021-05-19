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
def stage2_inputs_placeholder(input_feature_channels=241,
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

        # for i, layer_name in enumerate(sorted(rpn_params.keys())):
        #     roi_coors, roi_features, roi_num_list, _, _ = \
        #         point_conv(input_coors=coors,
        #                    input_features=features,
        #                    input_num_list=num_list,
        #                    voxel_idx=voxel_idx,
        #                    center_idx=center_idx,
        #                    layer_params=rpn_params[layer_name],
        #                    dimension_params=dimension_params,
        #                    grid_buffer_size=config.grid_buffer_size,
        #                    output_pooling_size=config.output_pooling_size,
        #                    scope="stage1_" + layer_name,
        #                    is_training=is_training,
        #                    trainable=trainable,
        #                    mem_saving=mem_saving,
        #                    model_params=model_params,
        #                    bn_decay=bn)

        roi_features = concat_features
        roi_coors = coors
        roi_num_list = num_list

        # roi_features = conv_1d(input_points=roi_features,
        #                        num_output_channels=256,
        #                        drop_rate=0.,
        #                        model_params=model_params,
        #                        scope='stage1_rpn_fc_0',
        #                        is_training=is_training,
        #                        trainable=trainable)

        roi_features = conv_1d(input_points=roi_features,
                               num_output_channels=256,
                               drop_rate=0.,
                               model_params=model_params,
                               scope='stage1_rpn_fc_1',
                               is_training=is_training,
                               trainable=trainable,
                               second_last_layer=True)

        roi_logits = conv_1d(input_points=roi_features,
                             num_output_channels=config.output_attr,
                             drop_rate=0.,
                             model_params=model_params,
                             scope='stage1_rpn_fc_2',
                             is_training=is_training,
                             trainable=trainable,
                             last_layer=True)

        roi_attrs = get_roi_attrs(input_logits=roi_logits,
                                  base_coors=roi_coors,
                                  anchor_size=anchor_size,
                                  is_eval=is_eval)

        roi_conf_logits = roi_logits[:, 7]

        if not trainable:
            roi_conf = tf.nn.sigmoid(roi_conf_logits)
            nms_idx = rotated_nms3d_idx(roi_attrs, roi_conf, nms_overlap_thresh=0.7, nms_conf_thres=0.7)
            roi_coors = tf.gather(roi_coors, nms_idx, axis=0)
            roi_attrs = tf.gather(roi_attrs, nms_idx, axis=0)
            roi_conf_logits = tf.gather(roi_conf_logits, nms_idx, axis=0)
            roi_num_list = tf.expand_dims(tf.shape(nms_idx)[0], axis=0)

        return coors, concat_features, num_list, roi_coors, roi_attrs, roi_conf_logits, roi_num_list


def stage2_model(coors,
                 features,
                 num_list,
                 roi_attrs,
                 roi_conf_logits,
                 roi_ious,
                 roi_num_list,
                 is_training,
                 trainable,
                 is_eval,
                 mem_saving,
                 bn):

    roi_conf = tf.nn.sigmoid(roi_conf_logits)
    with tf.variable_scope("stage2"):
        bbox_roi_attrs, bbox_num_list, bbox_idx = roi_filter(input_roi_attrs=roi_attrs,
                                                             input_roi_conf=roi_conf,
                                                             input_roi_ious=roi_ious,
                                                             input_num_list=roi_num_list,
                                                             conf_thres=config.roi_thres,
                                                             iou_thres=config.iou_thres,
                                                             max_length=config.max_length,
                                                             with_negative=not is_eval)

        bbox_voxels = la_roi_pooling_fast(input_coors=coors,
                                          input_features=features,
                                          roi_attrs=bbox_roi_attrs,
                                          input_num_list=num_list,
                                          roi_num_list=bbox_num_list,
                                          voxel_size=config.roi_voxel_size,
                                          grid_buffer_size=4,
                                          grid_buffer_resolution=[0.2, 0.2, 0.4], # TODO: Need to change this param.
                                          pooling_size=4,
                                          dimension=config.dimension_training,
                                          offset=config.offset_training)

        for i in range((config.roi_voxel_size - (config.roi_voxel_size + 1) % 2) // 2):
            bbox_voxels = conv_3d(input_voxels=bbox_voxels,
                                  layer_params=config.refine_params,
                                  scope="stage2_refine_conv_{}".format(i),
                                  is_training=is_training,
                                  trainable=trainable,
                                  model_params=model_params,
                                  mem_saving=mem_saving,
                                  bn_decay=bn)

        bbox_features = tf.reduce_mean(bbox_voxels, axis=[1])

        bbox_features = conv_1d(input_points=bbox_features,
                              num_output_channels=256,
                              drop_rate=0.,
                              model_params=model_params,
                              scope='stage2_refine_fc_0',
                              is_training=is_training,
                              trainable=trainable,
                              second_last_layer=True)

        bbox_logits = conv_1d(input_points=bbox_features,
                              num_output_channels=config.output_attr + 1 + 3,
                              drop_rate=0.,
                              model_params=model_params,
                              scope='stage2_refine_fc_1',
                              is_training=is_training,
                              trainable=trainable,
                              last_layer=True)

        bbox_attrs = get_bbox_attrs(input_logits=bbox_logits,
                                    input_roi_attrs=bbox_roi_attrs,
                                    is_eval=is_eval)

        bbox_conf_logits = bbox_logits[:, 7]
        bbox_dir_logits = bbox_logits[:, 8]
        bbox_cls_logits = bbox_logits[:, 9:]

    return bbox_attrs, bbox_conf_logits, bbox_dir_logits, bbox_cls_logits, bbox_num_list, bbox_idx

