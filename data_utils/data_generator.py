import zmq
import numpy as np
import time
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
import multiprocessing
import sys
import struct


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
                                      point_cloud[:, 1] > 7.899438643537479,
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
            intensity = feature_normalize(point_cloud[:, 4:5], method=300.)
            num_list = np.array([len(coors)])

            yield info_tag, coors, intensity, num_list



if __name__ == '__main__':
    SaiKungGenerator = DataGenerator(range_x=[-11., 11.],
                                     range_y=[-4.8, 11],
                                     range_z=[0.5, 3.1])
    output_coors, output_intensity, output_num_list = [], [], []
    for i in tqdm(range(1000)):
        _, coors, intensity, num_list = next(SaiKungGenerator.read_from_zmq())

        dimension = [22., 15.8, 2.6]
        offset = [11., 4.8, -0.5]

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