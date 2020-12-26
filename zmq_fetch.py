#
# Hello World client in Python
# Connects REQ socket to tcp://localhost:5555
# Sends "Hello" to server, expects "World" back
#
# INSTALLED VIA EASY_INSTALL
import zmq
import numpy as np
import time
from tqdm import tqdm
from copy import deepcopy
from point_viz.converter import PointvizConverter
Converter = PointvizConverter(home='/home/akk/threejs/MTR')


# 2.93 milliseconds

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method._name_.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method._name_, (te - ts) * 1000))
        return result

    return timed


context = zmq.Context()

# Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:5557")

def convert_threejs_coors(coors):
    assert len(coors.shape) == 2
    threejs_coors = np.zeros(shape=coors.shape)
    threejs_coors[:, 0] = coors[:, 1]
    threejs_coors[:, 1] = coors[:, 2]
    threejs_coors[:, 2] = coors[:, 0]
    return threejs_coors

def get_and_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_and(output, conditions[i])
    return output



def read_from_zmq():
    # Do 10 requests, waiting each time for a response
    for request in tqdm(range(1)):
        print("Sending request ", request, "...")
        # socket.send_string("Hello")

        message = socket.recv()
        point_cloud = np.frombuffer(message, dtype=np.float32)
        # print(point_cloud.shape)
        point_cloud = deepcopy(np.reshape(point_cloud, (-1, 9)))
        point_cloud[:, 2] *= -1
        keep_idx = get_and_sets([point_cloud[:, 0] > -5,
                                 point_cloud[:, 0] < 5,
                                 point_cloud[:, 1] > -3,
                                 point_cloud[:, 1] < 3,
                                 point_cloud[:, 2] > -1,
                                 point_cloud[:, 2] < 1])
        point_cloud = point_cloud[keep_idx, :]


        Converter.compile(task_name='test',
                          coors=convert_threejs_coors(point_cloud[:, :3]),
                          intensity=point_cloud[:, 3],
                          bbox_params=None)
        # print(point_cloud.shape)
        # print(point_cloud)

        # print("Received reply ", request, "[", message, "]")

if __name__ == '__main__':
    read_from_zmq()
