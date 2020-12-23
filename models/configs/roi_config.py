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

# dimension = [22., 15.8, 2.6]
# offset = [11., 4.8, -0.5]

range_x = [-5., 5.]
range_y = [-3, 3.]
range_z = [-1., 1.]

dimension = [range_x[1]-range_x[0],
             range_y[1]-range_y[0],
             range_z[1]-range_z[0]]

offset = [-range_x[0],
          -range_y[0],
          -range_z[0],]

local = False

gpu_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

model_file_name = os.path.basename(__file__).split('.')[0] + '.py'
model_file_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(model_file_dir, model_file_name)

bbox_padding = aug_config['nbbox']
batch_size = 16
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
cls_num = 4
output_attr = 8 + cls_num
training_epochs = 800

base_params = {'base_0': {'subsample_res': 0.05, 'c_out': 16, 'kernel_res': 0.05, 'padding': -1.},
               # 'base_1': {'subsample_res': 0.05, 'c_out': 16, 'kernel_res': 0.05, 'padding': 0.},
               # 'base_3': {'subsample_res': 0.1, 'c_out': 32, 'kernel_res': 0.1, 'padding': 0.},
               'base_4': {'subsample_res': 0.1, 'c_out': 32, 'kernel_res': 0.1, 'padding': 0.},
               # 'base_5': {'subsample_res': 0.2, 'c_out': 64, 'kernel_res': 0.2, 'padding': 0.},
               # 'base_6': {'subsample_res': 0.4, 'c_out': 128, 'kernel_res': 0.4, 'padding': 0.},
               'base_7': {'subsample_res': 0.1, 'c_out': 64, 'kernel_res': 0.2, 'padding': 0.},
               'base_8': {'subsample_res': 0.2, 'c_out': 128, 'kernel_res': 0.4, 'padding': 0.}}

rpn_params = {'subsample_res': 0.3, 'c_out': 128, 'kernel_res': 0.6, 'padding': 0.}
