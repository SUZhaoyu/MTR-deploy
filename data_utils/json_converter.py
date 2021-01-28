import numpy as np
import json

def convert_json(bboxes):
    json_dict = {}
    bbox_list = []
    cls_list = ["pedestrian", "luggages"]
    for i in range(len(bboxes)):
        bbox_dict = {}
        bbox = bboxes[i]
        w, l, h, x, y, z, r, cls = bbox[:8]
        if cls == 1:
            bbox_dict["center"] = {"x": float(x), "y": float(y), "z": float(z)}
            bbox_dict["width"] = float(w)
            bbox_dict["length"] = float(l)
            bbox_dict["height"] = float(h)
            bbox_dict["angle"] = float(r)
            bbox_dict["object_id"] = cls_list[int(cls)]
            bbox_list.append(bbox_dict)
    json_dict["bounding_boxes"] = bbox_list
    return json_dict
