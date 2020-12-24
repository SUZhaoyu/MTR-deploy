import numpy as np
from numpy import cos, sin
from shapely.geometry import Polygon
from copy import deepcopy

def box2vertex(box):
    w, l, h, x, y, z, r = box[:7]
    x_0, x_1 = x - w / 2., x + w / 2.
    y_0, y_1 = y - l / 2., y + l / 2.
    vertex = np.array([[x_0, y_0],
                       [x_0, y_1],
                       [x_1, y_1],
                       [x_1, y_0]])
    vertex -= [x, y]
    T_r = np.array([[cos(r), -sin(r)],
                    [sin(r), cos(r)]])
    vertex = np.dot(vertex, np.transpose(T_r))
    vertex += [x, y]
    return vertex

def intersect_height(box_a, box_b):
    ha_low, ha_high = box_a[5] - box_a[2] / 2., box_a[5] + box_a[2] / 2.
    hb_low, hb_high = box_b[5] - box_b[2] / 2., box_b[5] + box_b[2] / 2.
    if hb_low > ha_high or ha_low > hb_high:
        return 0.
    else:
        return min(ha_high, hb_high) - max(ha_low, hb_low)



def cal_bev_areas(box_a, box_b):
    a = Polygon(box2vertex(box_a))
    b = Polygon(box2vertex(box_b))
    return a.area, b.area, a.intersection(b).area

def cal_3d_iou(box_a, box_b):
    h_a, h_b = box_a[2], box_b[2]
    bev_area_a, bev_area_b, bev_intersect_area = cal_bev_areas(box_a, box_b)
    height = intersect_height(box_a, box_b)
    V_a = bev_area_a * h_a
    V_b = bev_area_b * h_b
    V_intersect = bev_intersect_area * height
    iou_3d = V_intersect / (V_a + V_b - V_intersect)
    if iou_3d >= 1:
        print(iou_3d)
    return iou_3d

def nms(bboxes, thres=0.2):
    bboxes_collection = []
    bboxes = bboxes[bboxes[:, -1].argsort()[::-1]]
    output_bboxes = []
    while len(bboxes) != 0:
        target_bbox = bboxes[0]
        delete_id = []
        current_group = [bboxes[0]]
        for i in np.arange(1, len(bboxes)):
            iou = cal_3d_iou(target_bbox, bboxes[i])
            if iou > thres:
                delete_id.append(i)
                current_group.append(bboxes[i])
        bboxes_collection.append(current_group)
        bboxes = np.delete(bboxes, delete_id, axis=0)
        output_bboxes.append(target_bbox)
        bboxes = np.delete(bboxes, 0, axis=0)
    return np.array(output_bboxes), bboxes_collection

def nms_average(bboxes_collection):
    output_bboxes = []
    for bboxes_group in bboxes_collection:
        group = np.array(bboxes_group)
        bboxes = np.zeros(group.shape[1])
        for i in range(7):
            bboxes[i] = np.percentile(group[:, i], 50)
        for i in np.arange(7, group.shape[1]):
            bboxes[i] = np.max(group[:, i])
        output_bboxes.append(bboxes)
    return np.array(output_bboxes)

def batch_box2vertex(bboxes):
    category_dict = ["Car", "Gogovan", "Minibus", "Bus", "Truck", "Motorbike"]
    vertex = []
    category = []
    for box in bboxes:
        vertex.append(box2vertex(box))
        category.append(category_dict[int(box[7])])
    return np.array(vertex), category

