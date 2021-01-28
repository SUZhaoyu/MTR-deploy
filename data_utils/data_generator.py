import zmq
import numpy as np
import time
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from point_viz.converter import PointvizConverter
import multiprocessing
import sys
from os.path import join
import struct

Converter = PointvizConverter(home='/home/akk/threejs/MTR')
from data_utils.normalization import convert_threejs_bbox_with_prob, convert_threejs_coors

# frame_id 4
# count 4
# lidar_ms 8
# unix_ms 8

sys.path.append("/home/akk/MTR-deploy")
from data_utils.normalization import feature_normalize
import logging


def get_and_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_and(output, conditions[i])
    return output


def get_or_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_or(output, conditions[i])
    return output

def get_rotation_matrix(rx, ry, rz):
    tr_x = np.array([[1, 0, 0],
                     [0, np.cos(rx), -np.sin(rx)],
                     [0, np.sin(rx), np.cos(rx)]])
    tr_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                     [0, 1, 0],
                     [-np.sin(ry), 0, np.cos(ry)]])
    tr_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz), np.cos(rz), 0],
                     [0, 0, 1]])
    tr_m = np.matmul(tr_z, np.matmul(tr_y, tr_x))
    return tr_m

tr_m = get_rotation_matrix(3.1172693, -0.0087579, 0.)


class DataGenerator(object):
    def __init__(self,
                 range_x,
                 range_y,
                 range_z,
                 max_queue=20,
                 tcp="tcp://localhost:5557"):
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.tcp = tcp
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 10)
        self.socket.connect(self.tcp)
        print("Listening zmq data from {}".format(self.tcp))

    def read_from_zmq(self):
        while True:
            message = self.socket.recv()
            frame_id = deepcopy(np.frombuffer(message[:4], dtype=np.int32))[0]
            point_count = deepcopy(np.frombuffer(message[4:8], dtype=np.int32))[0]
            lidar_timestamp = deepcopy(np.frombuffer(message[8:16], dtype=np.float64))[0]
            unix_timestamp = deepcopy(np.frombuffer(message[16:24], dtype=np.float64))[0]
            info_tag = [frame_id, point_count, lidar_timestamp, unix_timestamp]
            point_cloud = deepcopy(np.frombuffer(message[24:], dtype=np.float32))

            point_cloud = np.reshape(point_cloud, newshape=(-1, 9))
            if len(point_cloud) != point_count:
                logging.warning("Number of actual input point is different from buffer head ({} vs. {}) @ {}"
                                .format(len(point_cloud), point_count, datetime.fromtimestamp(unix_timestamp)))
            point_cloud[:, :3] *= [1., -1., -1]
            point_cloud[:, 2] += 3.3215672969818115
            # point_cloud[:, 2] *= -1

            ignore_idx_area_0 = get_and_sets([point_cloud[:, 0] > -6.949255957496236,
                                              point_cloud[:, 0] < 3.251014678502448,
                                              point_cloud[:, 1] > 7.5,
                                              point_cloud[:, 1] < 11.001353179972943])

            ignore_idx_area_1 = get_and_sets([point_cloud[:, 0] > -13.027740395930584,
                                              point_cloud[:, 0] < -9.21177287224803,
                                              point_cloud[:, 1] > 7.428958051420843,
                                              point_cloud[:, 1] < 10.554803788903929])

            ignore_idx = get_or_sets([ignore_idx_area_0, ignore_idx_area_1])

            keep_idx = get_and_sets([point_cloud[:, 0] > self.range_x[0],
                                     point_cloud[:, 0] < self.range_x[1],
                                     point_cloud[:, 1] > self.range_y[0],
                                     point_cloud[:, 1] < self.range_y[1],
                                     point_cloud[:, 2] > self.range_z[0],
                                     point_cloud[:, 2] < self.range_z[1],
                                     np.logical_not(ignore_idx)])

            # keep_idx = get_and_sets([point_cloud[:, 0] > self.range_x[0],
            #                          point_cloud[:, 0] < self.range_x[1],
            #                          point_cloud[:, 1] > self.range_y[0],
            #                          point_cloud[:, 1] < self.range_y[1],
            #                          point_cloud[:, 2] > self.range_z[0],
            #                          point_cloud[:, 2] < self.range_z[1]])

            point_cloud = point_cloud[keep_idx, :]
            coors = point_cloud[:, :3]

            coors[:, 2] -= 3.3215672969818115
            coors[:, :3] *= [1., -1., -1]
            coors[:, :3] = np.matmul(tr_m, np.transpose(coors[:, :3])).transpose()
            coors[:, 2] += 2.63
            intensity = feature_normalize(point_cloud[:, 4:5], method=300.)
            num_list = np.array([len(coors)])

            yield info_tag, coors, intensity, num_list


if __name__ == '__main__':
    SaiKungGenerator = DataGenerator(range_x=[-20., 11.],
                                     range_y=[-5., 12.],
                                     range_z=[0.5, 3.1])
    output_coors, output_intensity, output_num_list = [], [], []
    for i in tqdm(range(20000)):
        _, coors, intensity, num_list = next(SaiKungGenerator.read_from_zmq())

        # np.savetxt(join('/home/akk/data-viz/%06d.csv'%i), coors, delimiter=",")

        # Converter.compile(task_name="MTR_gen_filter",
        #                   coors=convert_threejs_coors(coors),
        #                   intensity=intensity[:, 0],
        #                   default_rgb=None)

        dimension = [32., 20., 4.0]
        offset = [20., 6., 0.5]

        coors += offset
        coors_min = np.min(coors, axis=0)
        coors_max = np.max(coors, axis=0)
        for j in range(3):
            if coors_min[j] < 0 or coors_max[j] > dimension[j]:
                print(coors_min, coors_max)

    #     if i % 100 == 0:
    #         output_coors.append(coors)
    #         output_intensity.append(intensity)
    #         output_num_list.append(num_list)
    # np.save("coors.npy", output_coors)
    # np.save("intensity.npy", output_intensity)
    # np.save("num_list.npy", output_num_list)
