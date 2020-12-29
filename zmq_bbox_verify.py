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


# 2.93 milliseconds


context = zmq.Context()

# Socket to talk to server
socket = context.socket(zmq.PULL)
socket.bind("tcp://127.0.0.1:5555")

if __name__ == '__main__':
    message = socket.recv()
    frame_id = deepcopy(np.frombuffer(message[:4], dtype=np.int32))[0]
    count = deepcopy(np.frombuffer(message[4:8], dtype=np.int32))[0]
    lidar_timestamp = deepcopy(np.frombuffer(message[8:16], dtype=np.float64))[0]
    unix_timestamp = deepcopy(np.frombuffer(message[16:24], dtype=np.float64))[0]
    info_tag = [frame_id, count, lidar_timestamp, unix_timestamp]
    point_cloud = deepcopy(np.frombuffer(message[24:], dtype=np.float32))

    point_cloud = np.reshape(point_cloud, newshape=(-1, 9))
    print(" ")


