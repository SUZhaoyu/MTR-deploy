import os

import numpy as np

aug_config = {'nbbox': 128,
              'rotate_range': 0,
              'rotate_mode': 'u',
              'scale_range': 0.05,
              'scale_mode': 'u',
              'drop_out': 0.1,
              'flip': False,
              'shuffle': True,
              'paste_augmentation': True,
              'paste_instance_num': 64,
              'maximum_interior_points': 20,
              'normalization': None}

# range_x = [-32., 10.9]
range_x = [-32., 8.]
range_y = [-15.9, 10.8]
# range_y = [-4., 10.8]
range_z = [-0.5, 2.1]

dimension_training = [50., 30., 6.]
offset_training = [35., 15., 2.]

# dimension_training = [100., 100., 9.]
# offset_training = [10., 10., 5.]

# dimension_training = [72, 80.0, 4.]
# offset_training = [2., 40.0, 3.]

anchor_size = [0.6, 0.6, 1.6]
grid_buffer_size = 3
output_pooling_size = 5

diff_thres = 3
cls_thres = 2

local = False

gpu_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

model_file_name = os.path.basename(__file__).split('.')[0] + '.py'
model_file_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(model_file_dir, model_file_name)

bbox_padding = aug_config['nbbox']
batch_size = 4
decay_epochs = 5

init_lr_stage1 = 1e-3
lr_scale_stage1 = True

init_lr_stage2 = 2e-4
lr_scale_stage2 = False

lr_decay = 0.5
lr_warm_up = True
cls_loss_scale = 1.
weight_decay = 5e-4
valid_interval = 5
use_trimmed_foreground = False
paste_augmentation = False
use_la_pooling = False
norm_angle = False
xavier = False
stddev = 1e-3
activation = 'relu'
normalization = None
num_worker = 6
weighted = False
use_l2 = True
output_attr = 8
# stage1_training_epoch = 25
total_epoch = 300

roi_thres = 0.7
iou_thres = 0.3
max_length = 512
roi_voxel_size = 5

# base_params_inference = {'base_0': {'subsample_res': 0.05, 'c_out':  16, 'kernel_res': 0.05, 'concat': False},
#                          'base_1': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'concat': True},
#                          'base_2': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'concat': True},
#                          'base_3': {'subsample_res': 0.30, 'c_out':  64, 'kernel_res': [0.20, 0.20, 0.40], 'concat': True},
#                          'base_4': {'subsample_res': None, 'c_out': 128, 'kernel_res': [0.40, 0.40, 0.60], 'concat': True}}

base_params_inference = {'base_0': {'subsample_res': 0.05, 'c_out':  16, 'kernel_res': 0.05, 'concat': True},
                         'base_1': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'concat': False},
                         'base_2': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.10, 'concat': True},
                         'base_3': {'subsample_res': 0.20, 'c_out':  32, 'kernel_res': 0.20, 'concat': False},
                         'base_4': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.20, 'concat': True},
                         'base_5': {'subsample_res': 0.30, 'c_out':  64, 'kernel_res': [0.20, 0.20, 0.40], 'concat': False},
                         'base_6': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.20, 0.20, 0.40], 'concat': True},
                         'base_7': {'subsample_res': None, 'c_out': 128, 'kernel_res': [0.40, 0.40, 0.60], 'concat': False},
                         'base_8': {'subsample_res': None, 'c_out': 128, 'kernel_res': [0.40, 0.40, 0.60], 'concat': True}}


refine_params = {'c_out': 256, 'kernel_size': 3, 'padding': 0.}