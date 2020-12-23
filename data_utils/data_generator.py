import zmq
import numpy as np
import time
from tqdm import tqdm
from copy import deepcopy
import multiprocessing
from data_utils.normalization import feature_normalize
import logging

def get_and_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_and(output, conditions[i])
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
            point_cloud[:, 2] *= -1
            keep_idx = get_and_sets([point_cloud[:, 0] > self.range_x[0],
                                     point_cloud[:, 0] < self.range_x[1],
                                     point_cloud[:, 1] > self.range_y[0],
                                     point_cloud[:, 1] < self.range_y[1],
                                     point_cloud[:, 2] > self.range_z[0],
                                     point_cloud[:, 2] < self.range_z[1]])
            point_cloud = point_cloud[keep_idx, :]
            coors = point_cloud[:, :3]
            intensity = feature_normalize(point_cloud[:, 4:5], method=300.)
            num_list = np.array([len(coors)])

            yield coors, intensity, num_list



if __name__ == '__main__':
    SaiKungGenerator = DataGenerator(range_x=[-5., 5.],
                                     range_y=[-3., 3.],
                                     range_z=[-1., 1.])
    for _ in tqdm(range(10000)):
        coors, intenrity, num_list = next(SaiKungGenerator.read_from_zmq())
    print(coors.shape)

