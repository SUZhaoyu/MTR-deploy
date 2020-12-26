import zmq
import numpy as np
import time
from tqdm import tqdm
from copy import deepcopy
import multiprocessing
import sys


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
            point_cloud = deepcopy(np.frombuffer(message, dtype=np.float32))

            point_cloud = np.reshape(point_cloud, newshape=(-1, 9))
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

            yield coors, intensity, num_list



if __name__ == '__main__':
    SaiKungGenerator = DataGenerator(range_x = [-11., 11.],
                                    range_y = [-4.8, 11],
                                    range_z = [0.5, 3.1])
    for _ in tqdm(range(10000)):
        coors, intenrity, num_list = next(SaiKungGenerator.read_from_zmq())
        print(coors.shape)

