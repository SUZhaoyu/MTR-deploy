import tensorflow as tf

from models.utils.iou_utils import cal_3d_iou, get_roi_attrs_from_logits
from models.utils.model_layers import point_conv, fully_connected
from models.tf_ops.custom_ops import roi_logits_to_attrs

anchor_size = [0.6, 0.6, 1.6]
eps = tf.constant(1e-6)


def inputs_placeholder(input_channels=1,
                       bbox_padding=128):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, None], name='input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[None], name='input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[None, bbox_padding, 8], name='input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p


def model(input_coors,
          input_features,
          input_num_list,
          is_training,
          config,
          bn):
    model_params = {'xavier': config.xavier,
                    'stddev': config.stddev,
                    'activation': config.activation,
                    'dimension': config.dimension,
                    'offset': config.offset}
    base_params = config.base_params

    with tf.variable_scope("rcnn"):
        coors, features, num_list = input_coors, input_features, input_num_list
        for layer_name in sorted(base_params.keys()):
            coors, features, num_list = point_conv(input_coors=coors,
                                                   input_features=features,
                                                   input_num_list=num_list,
                                                   layer_params=base_params[layer_name],
                                                   scope=layer_name,
                                                   is_training=is_training,
                                                   model_params=model_params,
                                                   bn_decay=bn)

        base_coors, roi_features, roi_num_list = point_conv(input_coors=coors,
                                                            input_features=features,
                                                            input_num_list=num_list,
                                                            layer_params=config.rpn_params,
                                                            scope="rpn_conv",
                                                            is_training=is_training,
                                                            model_params=model_params,
                                                            bn_decay=bn)

        roi_logits = fully_connected(input_points=roi_features,
                                     num_output_channels=config.output_attr,
                                     drop_rate=0.,
                                     model_params=model_params,
                                     scope='rpn_fc',
                                     is_training=is_training,
                                     last_layer=True)

        roi_attrs = roi_logits_to_attrs(base_coors=base_coors,
                                        input_logits=roi_logits,
                                        anchor_size=anchor_size)
        roi_conf_logits = roi_logits[:, 7]
        roi_cls_logits = roi_logits[:, 8:]

        return base_coors, roi_attrs, roi_num_list, roi_conf_logits, roi_cls_logits

