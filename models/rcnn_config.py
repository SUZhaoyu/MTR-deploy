import os

import numpy as np

aug_config = {'nbbox': 128,
              'rotate_range': np.pi * 2,
              'rotate_mode': 'u',
              'scale_range': 0.05,
              'scale_mode': 'u',
              'drop_out': 0.1,
              'flip': False,
              'shuffle': True,
              'paste_augmentation': True,
              'paste_instance_num': 16,
              'maximum_interior_points': 20,
              'normalization': '300'}

# range_x = [-11., 11.]
# range_y = [-4.8, 11]
# range_z = [0.5, 3.1]

# FIXME: The range is based on the old coor sys
range_x = [-20., 11.]
range_y = [-5., 12.]
range_z = [0.5, 3.1]

# range_x = [-15., 11.]
# range_y = [-7.5, 6.]
# range_z = [0.5, 3.1]

# dimension = [31., 17., 2.6]
# offset = [20., 5., -0.5]
# FIXME: The dimension and offset is based on the new coor sys
dimension = [32., 20., 4.0]
offset = [20., 6., 0.5]

# dimension = [36., 36., 4.]
# offset = [18., 18., 0.]

anchor_size = [0.8, 0.8, 1.6]

local = False

gpu_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

model_file_name = os.path.basename(__file__).split('.')[0] + '.py'
model_file_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(model_file_dir, model_file_name)

bbox_padding = aug_config['nbbox']
batch_size = 4
decay_epochs = 20
init_lr = 1e-4
lr_decay = 0.5
lr_scale = True
lr_warm_up = False
cls_loss_scale = 1.
weight_decay = 1e-3
valid_interval = 5
use_trimmed_foreground = False
paste_augmentation = True
use_la_pooling = False
norm_angle = False
xavier = True
stddev = 1e-3
activation = 'relu'
num_worker = 3
weighted = False
use_l2 = True
cls_num = 2
output_attr = 8 + cls_num
total_epoch = 800

roi_thres = 0.5
max_roi_per_instance = 300
roi_voxel_size = 5

base_params = {'base_0': {'subsample_res': 0.05, 'c_out':  16, 'kernel_res': 0.05, 'padding': -1.},
               'base_1': {'subsample_res': 0.10, 'c_out':  32, 'kernel_res': 0.10, 'padding':  0.},
               'base_2': {'subsample_res': 0.15, 'c_out':  64, 'kernel_res': 0.20, 'padding':  0.},
               'base_3': {'subsample_res': 0.20, 'c_out': 128, 'kernel_res': 0.40, 'padding':  0.}}

rpn_params = {'subsample_res': 0.30, 'c_out': 256, 'kernel_res': 0.60, 'padding': 0.}
refine_params = {'c_out': 256, 'kernel_size': 3, 'padding': 0.}

# base_params = {'base_0': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'padding': -1.},
#                'base_1': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'padding': 0.},
#                'base_2': {'subsample_res': 0.30, 'c_out':  64, 'kernel_res': 0.30, 'padding': 0.},
#                'base_3': {'subsample_res': 0.40, 'c_out': 128, 'kernel_res': 0.40, 'padding': 0.}}
#
# rpn_params = {'subsample_res': 0.60, 'c_out': 256, 'kernel_res': 0.60, 'padding': 0.}
# refine_params = {'c_out': 256, 'kernel_size': 3, 'padding': 0.}