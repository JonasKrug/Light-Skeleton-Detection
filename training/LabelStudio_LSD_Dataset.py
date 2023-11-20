# ...
import os
import json

# ...
import numpy as np

# ...
import cv2
import albumentations

# ...
import torch
from torch.utils.data import Dataset

from torchvision.transforms import functional



# ...
def collate_fn(batch):
    return tuple(zip(*batch))

# ...
def transfer_data_from_dict_to_dict(src_dict : dict, dst_dict : dict, dst_src_field_dict : dict) -> dict:

    for dst_field, src_field in dst_src_field_dict.items():

        if type(src_field) is type:
            dst_dict[dst_field] = src_field()

        elif callable(src_field):
            dst_dict[dst_field] = src_field(src_dict)

        else:
            dst_dict[dst_field] = src_dict[src_field] 

    return dst_dict

# ...
def get_leaf_and_non_leaf_children(parent_child_connections):

    non_leaf_nodes, leaf_nodes = list(), list()
    for kk, vv in parent_child_connections.items():

        if len(vv["children_ids"]) != 0:
            non_leaf_nodes.append([kk, vv])

        else:
            leaf_nodes.append([kk, vv])

    return leaf_nodes, non_leaf_nodes

# ...
def parse_LS_image_annotation_result_dict(result_dict : dict) -> dict:

    dst_dict = dict()
    dst_result_field_dict = {"type": "type", "origin": "origin", "image_rotation": "image_rotation", 
                             "image_shape": lambda dict_: [dict_["original_width"], dict_["original_height"]], "data": "value", "children": dict}

    return transfer_data_from_dict_to_dict(result_dict, dst_dict, dst_result_field_dict)

# ...
def parse_LS_image_task_annotation(task_dict : dict):

    # ...
    meta_info = dict()
    meta_task_field_dict = {"id": "id", 
                            "image_path": lambda dict_: dict_["data"]["image"], 
                            "nr_annotations": lambda dict_: dict_["total_annotations"] - dict_["cancelled_annotations"], "relations": dict}
    
    meta_info = transfer_data_from_dict_to_dict(task_dict, meta_info, meta_task_field_dict)

    # ...
    annotations_info = list()

    for json_annotation in task_dict["annotations"]:

        if json_annotation["was_cancelled"]:
            continue

        # ...
        tmp_annotation = dict()
        tmp_annotation["id"] = json_annotation["unique_id"]

        meta_info["relations"][tmp_annotation["id"]] = list()

        # ...
        relation_idx = [idx for idx, elem in enumerate(json_annotation["result"]) if elem["type"] == "relation"]
        for index in sorted(relation_idx, reverse=True):
            
            relation_annotation = json_annotation["result"][index]
            del relation_annotation["type"]

            meta_info["relations"][tmp_annotation["id"]].append(relation_annotation)
            del json_annotation["result"][index]

        # ...
        tmp_annotation["result_ids"] = [json_result["id"] for json_result in json_annotation["result"]]
        
        # ...
        parent_child_connections = {json_result["id"]: {"parent": None, "children_ids": [], "data": parse_LS_image_annotation_result_dict(json_result)} for json_result in json_annotation["result"]}
        for json_result in json_annotation["result"]:

            is_child = ("parentID" in json_result.keys()) and (json_result["parentID"] is not None)
            if is_child:

                parent_child_connections[json_result["id"]]["parent"] = json_result["parentID"]
                parent_child_connections[json_result["parentID"]]["children_ids"].append(json_result["id"])            

        # ...
        leaf_nodes, non_leaf_nodes = get_leaf_and_non_leaf_children(parent_child_connections)

        # ...
        while len(non_leaf_nodes) != 0:

            for leaf in leaf_nodes:

                parent_node = leaf[1]["parent"]
                if parent_node is None:
                    pass

                else:

                    parent_child_connections[parent_node]["data"]["children"][leaf[0]] = leaf[1]["data"]
                    del parent_child_connections[leaf[0]]
                    parent_child_connections[parent_node]["children_ids"].remove(leaf[0]) 

            leaf_nodes, non_leaf_nodes = get_leaf_and_non_leaf_children(parent_child_connections)

        # ...
        annotations_info.append({kk: vv["data"] for kk, vv in parent_child_connections.items()})    

    return meta_info, annotations_info

# ...
def parse_LS_image_annotation(file_path : str):

    # ...
    with open(file_path, "r") as fp:
        content = json.load(fp)

    # ...
    meta_info, annotation_info = [], []
    for task_dict in content:

        parsed_meta_info, parsed_annotation_info = parse_LS_image_task_annotation(task_dict)

        meta_info.append(parsed_meta_info)
        annotation_info.append(parsed_annotation_info)

    return meta_info, annotation_info


class LS2BlinkerDetectionDataset(Dataset):
    
    def __init__(self, dataset_json_path : str, 
                       path_to_project_image_dir : str,
                       kp_label_list : list = None,
                       bbox_label_list : list = None,
                       bbox_label_to_blinker_state_func = None,
                       transform_func = None, 
                       demo : bool = False):
        
        # ...:
        dataset_meta, dataset_data = parse_LS_image_annotation(dataset_json_path)

        self.dataset = []
        for meta, data in zip(dataset_meta, dataset_data):

            for annotations in data:

                temp_datapoint = dict()

                temp_datapoint["id"] = meta["id"]
                temp_datapoint["image_path"] = os.path.join(path_to_project_image_dir, "/".join(meta["image_path"].split("/")[-2:])) #  pathlib.PurePath(meta["image_path"]).name)

                temp_datapoint["annotations"] = annotations
                temp_datapoint["image_id"] = meta["id"]

                self.dataset.append(temp_datapoint)

        # ...
        self.kp_label_list = self._get_default_kp_labellist() if kp_label_list is None else kp_label_list
        self.bbox_label_list = self._get_default_bbox_labellist() if bbox_label_list is None else bbox_label_list

        self.bbox_label_to_blinker_state_func = self._get_default_bbox_label_to_blinker_state_func() if bbox_label_to_blinker_state_func is None else bbox_label_to_blinker_state_func

        # ...
        self.transform = self._transform() if transform_func is not None else None
        self.demo = demo

    # ...
    def _get_default_kp_labellist(self):
        return ["Left Front Light", "Left Mirror", "Right Front Light", "Right Mirror", "Right Rear Light", "Center Rear Light", "Left Rear Light"]

    def _get_default_bbox_labellist(self):
        return ["Background", "Brake", "Brake + Hazard", "Brake + Left Blink", "Brake + Right Blink", "Hazard", "Left Blink", "Disabled", "Right Blink"]

    def _get_default_bbox_label_to_blinker_state_func(self):

        states_for_label = {
            "Disabled": (0, 0, 0),
            "Brake": (1, 0, 0), "Left Blink": (0, 1, 0), "Right Blink": (0, 0, 1),
            "Brake + Left Blink": (1, 1, 0), "Brake + Right Blink": (1, 0, 1), "Hazard": (0, 1, 1),
            "Brake + Hazard": (1, 1, 1)}
        
        return lambda bbox_label: [[[0, l], [0, l], [0, r], [0, r], [b, r], [b, 0], [b, l]]
                                   for (b, l, r) in [states_for_label[bbox_label]]][0]

    # ...
    def _load_image(self, img_path):

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)  
        return img_original

    def _transform(self):
        return albumentations.Compose(
                    [albumentations.Sequential([], p=1)],
                    keypoint_params = albumentations.KeypointParams(format='xy'), # Keypoint Format (see https://albumentations.ai/docs/getting_started/keypoints_augmentation/)
                    bbox_params = albumentations.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])) # Bboxes w/ Labels Format (see https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)
                
    # ...


    # ...
    def __getitem__(self, idx):

        # ...
        datapoint = self.dataset[idx]

        # TODO: Remove the Replacement Function;
        image_path = datapoint["image_path"]
        image_original = self._load_image(image_path)

        # ...
        bboxes_original = np.array([[
                            round(dp["data"]["x"] * (dp["image_shape"][0]/100)), 
                            round(dp["data"]["y"] * (dp["image_shape"][1]/100)),
                            round((dp["data"]["x"] + dp["data"]["width"]) * (dp["image_shape"][0]/100)),
                            round((dp["data"]["y"] + dp["data"]["height"]) * (dp["image_shape"][1]/100))
                            ] for dp in datapoint["annotations"].values()])
        
        bbox_label2position = {label: idx for idx, label in enumerate(self.bbox_label_list)}
        bboxes_labels_original = np.array([bbox_label2position[dp["data"]["rectanglelabels"][0]] for dp in datapoint["annotations"].values()])

        # ...
        keypoints_label2position = {label: idx for idx, label in enumerate(self.kp_label_list)}
        keypoints_original = np.zeros((bboxes_original.shape[0], len(self.kp_label_list), 3))

        for bbox_idx, dp in enumerate(datapoint["annotations"].values()):
            for kp in dp["children"].values():

                label_idx = keypoints_label2position[kp["data"]["bboxkeypointlabels"][0]]
                keypoints_original[bbox_idx][label_idx] = [kp["data"]["x"] * (kp["image_shape"][0]/100), kp["data"]["y"] * (kp["image_shape"][1]/100), 1]
        
        # ...
        if self.transform:   

            # ...
            keypoints_original_flattened = keypoints_original[:,:,:2].reshape((-1, 2)) 
            
            # ...
            transformed = self.transform(image = image_original, 
                                         bboxes = bboxes_original, 
                                         bboxes_labels = bboxes_labels_original, 
                                         keypoints = keypoints_original_flattened)
            
            image = transformed['image']
            bboxes = transformed['bboxes']

            keypoints = keypoints_original.copy()
            keypoints[:,:,:-1] = np.reshape(transformed['keypoints'], (keypoints.shape[0], keypoints.shape[1], 2))

        # ...
        else:
            image, bboxes, keypoints = image_original, bboxes_original, keypoints_original        

        # ...     
        target = {"image_id": datapoint["image_id"],
                  "boxes": torch.as_tensor(bboxes, dtype = torch.float32),
                  "labels": torch.as_tensor(bboxes_labels_original, dtype = torch.int64), 
                  "keypoints": torch.as_tensor(keypoints, dtype = torch.float32), 
                  
                  "keypoint_states": torch.as_tensor([
                                                self.bbox_label_to_blinker_state_func(self.bbox_label_list[bbox_label]) 
                                                for bbox_label in bboxes_labels_original], dtype=torch.float32)}   

        image = functional.to_tensor(image)
    
        # ...
        if self.demo:

            image_original = functional.to_tensor(image_original)
            target_original = { "image_id": datapoint["image_id"],
                                "boxes": torch.as_tensor(bboxes_original, dtype = torch.float32),
                                "labels": torch.as_tensor(bboxes_labels_original, dtype = torch.int64), 
                                "keypoints": torch.as_tensor(keypoints_original, dtype = torch.float32), 
                    
                                "keypoint_states": torch.as_tensor([
                                                    self.bbox_label_to_blinker_state_func(self.bbox_label_list[bbox_label]) 
                                                    for bbox_label in bboxes_labels_original], dtype=torch.float32)} 
    
            return image, target, image_original, target_original    

        return image, target
    
    def __len__(self):
        return len(self.dataset)

