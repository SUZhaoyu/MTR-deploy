import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    data_concat = []
    for i in tqdm(range(150)):
        data = np.fromfile("/home/akk/calib/lidar1_%05d.pcd" % int(i+5), dtype=np.float32)
        data = np.reshape(data, [-1, 9])

        coors_sum = data[:, 0] + data[:, 1] + data[:, 2]
        valid_idx = coors_sum != 0
        data = data[valid_idx, 4]
        data_concat.append(data)
        print(len(data))
    data_concat = np.concatenate(data_concat)
    print(data_concat.shape)
    print(np.mean(data_concat), np.std(data_concat))

# lidat_0: mean: 20.521528 std: 33.33701 # new, first half
# lidat_1: mean: 38.916435 std: 55.364727 # old, second half
