import numpy as np
import tensorflow as tf
from models.tf_ops.custom_ops import raw_data_filtering
from models.tf_ops.test.test_utils import plot_points
from tqdm import tqdm

raw_data = np.loadtxt("Calb0.csv", delimiter=",")
print(raw_data.shape)
col, row, attrs = 2048, 64, 9
raw_data_p = tf.placeholder(dtype=tf.float32, shape=[col*row, attrs])

data = raw_data_filtering(input_data=raw_data_p)
data = tf.reduce_mean(data)

config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.allow_soft_placement = False
config.log_device_placement = False
config.gpu_options.visible_device_list = '0'


if __name__ == '__main__':

    with tf.Session(config=config) as sess:
        for _ in tqdm(range(10000)):
            output_data = sess.run(data,
                                   feed_dict={raw_data_p: raw_data})
        # output_data = []
        # for i in tqdm(range(len(raw_data))):
        #     if abs(raw_data[i, 0]) + abs(raw_data[i, 1]) + abs(raw_data[i, 2]) > 0:
        #         if raw_data[i, 8] < 2000:
        #             if raw_data[i, 3] < raw_data[i, 7] - 4*raw_data[i, 8] or raw_data[i, 3] > raw_data[i, 7] + 4*raw_data[i, 8]:
        #                 output_data.append(raw_data[i])
        #                 print(i)
        #         else:
        #             output_data.append(raw_data[i])
        #             print(i)
        # output_data = np.stack(output_data, axis=0)
        # plot_points(output_data[:, :3], name='point_filtering')
        # print(" ")