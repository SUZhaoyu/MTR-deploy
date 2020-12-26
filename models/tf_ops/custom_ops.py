import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

grid_sampling_exe = tf.load_op_library(join(CWD, 'build', 'grid_sampling.so'))
print("******************** GridSamplingOp ***********************")
def grid_sampling(input_coors,
                  input_num_list,
                  resolution,
                  dimension=[70.4, 80.0, 4.0],
                  offset=[0., 40.0, 3.0]):
    output_idx, output_num_list = grid_sampling_exe.grid_sampling_op(input_coors=input_coors + offset,
                                                                     input_num_list=input_num_list,
                                                                     dimension=dimension,
                                                                     resolution=resolution)
    output_coors = tf.gather(input_coors, output_idx, axis=0)
    return output_coors, output_num_list


ops.NoGradient("GridSamplingOp")



# =============================================Voxel Sampling===============================================
voxel_sampling_exe = tf.load_op_library(join(CWD, 'build', 'voxel_sampling.so'))


def voxel_sampling(input_coors,
                   input_features,
                   input_num_list,
                   center_coors,
                   center_num_list,
                   resolution,
                   padding=0.,
                   dimension=[70.4, 80.0, 4.0],
                   offset=[0., 40.0, 3.0]):
    output_voxels, _ = voxel_sampling_exe.voxel_sampling_op(input_coors=input_coors + offset,
                                                            input_features=input_features,
                                                            input_num_list=input_num_list,
                                                            center_coors=center_coors + offset,
                                                            center_num_list=center_num_list,
                                                            dimension=dimension,
                                                            resolution=resolution,
                                                            padding_value=padding)
    return output_voxels


@ops.RegisterGradient("VoxelSamplingOp")
def voxel_sampling_grad(op, grad, _):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    input_features_grad = voxel_sampling_exe.voxel_sampling_grad_op(output_idx=output_idx,
                                                                    input_features=input_features,
                                                                    output_features_grad=grad)
    return [None, input_features_grad, None, None, None]


# =============================================Roi Logits To Attrs===============================================

roi_logits_to_attrs_exe = tf.load_op_library(join(CWD, 'build', 'roi_logits_to_attrs.so'))
def roi_logits_to_attrs(base_coors, input_logits, anchor_size):
    output_attrs = roi_logits_to_attrs_exe.roi_logits_to_attrs_op(base_coors=base_coors,
                                                                  input_logits=input_logits,
                                                                  anchor_size=anchor_size)
    return output_attrs
ops.NoGradient("RoiLogitsToAttrs")