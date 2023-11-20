import os
import json
import time
import numpy as np
from PIL import Image

keypoint_names = ["Left Front Light", "Left Mirror", "Right Front Light", "Right Mirror", "Right Rear Light", "Center Rear Light", "Left Rear Light"]
bbox_labels =  ["Left Blink", "Right Blink", "Hazard", "Brake + Left Blink", "Brake + Right Blink", "Brake + Hazard", "Brake", "Disabled"]

def get_light_center(top_left, top_right, bottom_left, bottom_right):
    if top_left[2] == 2 and top_right[2] == 2 and bottom_left[2] == 2 and bottom_right[2] == 2:
        center_keypoint = np.mean([top_left, top_right, bottom_left, bottom_right], axis=0)
        center_keypoint[2] = 1
        return center_keypoint
    elif top_left[2] == 2 and bottom_right[2] == 2:
        center_keypoint = np.mean([top_left, bottom_right], axis=0)
        center_keypoint[2] = 1
        return center_keypoint
    elif top_right[2] == 2 and bottom_left[2] == 2:
        center_keypoint = np.mean([top_right, bottom_left], axis=0)
        center_keypoint[2] = 1
        return center_keypoint
    else:
        return [0, 0, 0]

def convert_LSD_annotations_to_labelstudio_predictions(annotations, image_width, image_height):
    result = []
    for annotation in annotations:
        # Append bbox
        bbox_id = str(annotation['image_id']) + "-" + str(annotation['id'])
        if annotation['bbox'][0] != 0:
            bbox_x = (annotation['bbox'][0]/image_width)*100
        else:
            bbox_x = 0
        if annotation['bbox'][1] != 0:
            bbox_y = (annotation['bbox'][1]/image_height)*100
        else:
            bbox_y = 0
        if annotation['bbox'][2] != 0:
            bbox_width = (annotation['bbox'][2]/image_width)*100
        else:
            bbox_width = 0
        if annotation['bbox'][3] != 0:
            bbox_height = (annotation['bbox'][3]/image_height)*100
        else:
            bbox_height = 0
        result.append({
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
              "x": bbox_x,
              "y": bbox_y,
              "width": bbox_width,
              "height": bbox_height,
              "rotation": 0,
              "rectanglelabels": [
                "Disabled"
              ]
            },
            "id": bbox_id,
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "apollo3d_conversion",
            "score": 1
            })

        # Append Keypoints
        for i, keypoint in enumerate(annotation['keypoints']):
            if keypoint[2] == 1:
                if keypoint[0] != 0:
                    x = (keypoint[0]/image_width)*100
                else:
                    x = 0
                if keypoint[1] !=0:
                    y = (keypoint[1]/image_height)*100
                else:
                    y = 0
                result.append({
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "value": {
                      "x": x,
                      "y": y,
                      "width": 0.15,
                      "bboxkeypointlabels": [
                        keypoint_names[i]
                      ]
                    },
                    "parentID": bbox_id,
                    "id": bbox_id + str(i),
                    "from_name": "bkp-1",
                    "to_name": "image",
                    "type": "bboxkeypointlabels",
                    "origin": "apollo3d_conversion",
                    "score": 1
                    })          

    if len(annotations) > 0:
        return [{"id": str(annotations[0]['image_id']) + "_00", "model_version": "apollo3d", "result": result, "was_cancelled": False, "unique_id": str(annotations[0]['image_id']) + "_" + str(int(time.time()))}]
    else:
        return []

def convert_apollo_annotations_to_LSD(apollocar3d_annotation):
    LSD_annotations = []
    for annotation in apollocar3d_annotation:
        if annotation['iscrowd'] == 1: # Ignore crowd annotations 
            continue
        if annotation['num_keypoints'] > 0:
            keypoints = annotation['keypoints']
            keypoints = [keypoints[i:i+3] for i in range(0, len(keypoints), 3)]
            
            ## Merge Lights:
            # Font left:
            # top_left_c_left_front_car_light: 0 
            # top_right_c_left_front_car_light: 2
            # bottom_left_c_left_front_car_light: 1
            # bottom_right_c_left_front_car_light: 3
            left_front_light = get_light_center(keypoints[0], keypoints[2], keypoints[1], keypoints[3])

            # Font right:
            # top_left_c_right_front_car_light: 55
            # top_right_c_right_front_car_light: 57
            # bottom_left_c_right_front_car_light: 54
            # bottom_right_c_right_front_car_light: 56
            right_front_light = get_light_center(keypoints[55], keypoints[57], keypoints[54], keypoints[56])

            # Rear left:
            # top_left_c_left_rear_car_light: 22
            # top_right_c_left_rear_car_light: 25
            # bottom_left_c_left_rear_car_light: 23
            # bottom_right_c_left_rear_car_light: 26
            left_rear_light = get_light_center(keypoints[22], keypoints[25], keypoints[23], keypoints[26])

            # Rear right:
            # top_left_c_right_rear_car_light: 32
            # top_right_c_right_rear_car_light: 35
            # bottom_left_c_right_rear_car_light: 31
            # bottom_right_c_right_rear_car_light: 34
            right_rear_light = get_light_center(keypoints[32], keypoints[35], keypoints[31], keypoints[34])

            ## Get Mirrors:
            # top_left_c_left_front_door: 9
            if keypoints[9][2] == 2:
                left_mirror = keypoints[9]
                left_mirror[2] = 1
            else:
                left_mirror = [0, 0, 0]
            # top_right_c_right_front_car_door: 48
            if keypoints[48][2] == 2:
                right_mirror = keypoints[48]
                right_mirror[2] = 1
            else:
                right_mirror = [0, 0, 0]

            ## Get Center Rear Lights:
            # top_left_c_rear_glass: 24
            # top_right_c_rear_glass: 33
            if keypoints[24][2] == 2 and keypoints[33][2] == 2:
                center_rear_light = np.mean([keypoints[24], keypoints[33]], axis=0)
                center_rear_light[2] = 1
            else:
                center_rear_light = [0, 0, 0]

            LSD_annotations.append({'id': annotation['id'], 'image_id': annotation['image_id'], 'bbox': annotation['bbox'], 'keypoints': [left_front_light, left_mirror, right_front_light, right_mirror, right_rear_light, center_rear_light, left_rear_light]})
        else:
            LSD_annotations.append({'id': annotation['id'], 'image_id': annotation['image_id'], 'bbox': annotation['bbox'], 'keypoints': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]})
    
    return LSD_annotations

def load_apollo_3d(apollo_annotation_file):
    return json.load(open(apollo_annotation_file))

def point_in_image(point_x, point_y,  x_start, y_start, image_width, image_height):
    return point_x > x_start and point_y > y_start and point_x < x_start + image_width and point_y < y_start + image_height

def bbox_in_image(bbox, x_start, y_start, image_width, image_height):
    return point_in_image(bbox[0], bbox[1], x_start, y_start, image_width, image_height)  or point_in_image(bbox[0], bbox[1] + bbox[3], x_start, y_start, image_width, image_height) or point_in_image(bbox[0] + bbox[2], bbox[1], x_start, y_start, image_width, image_height) or point_in_image(bbox[0] + bbox[2], bbox[1] + bbox[3], x_start, y_start, image_width, image_height)

def adjust_LSD_annotations_to_cropped_image(LSD_annotations, x_start, y_start, cropped_image_width, cropped_image_height):
    LSD_annotations_cropped = []

    for annotation in LSD_annotations:
        if not bbox_in_image(annotation['bbox'], x_start, y_start, cropped_image_width, cropped_image_height): # Ignore annotations that are not in cropped image
            continue
        
        # Adjust bbox origin x
        annotation['bbox'][0] -= x_start
        bbox_0_old = annotation['bbox'][0]
        annotation['bbox'][0] = min(max(0, annotation['bbox'][0]), cropped_image_width)
        
        # Adjust bbox origin y
        annotation['bbox'][1] -= y_start
        bbox_1_old = annotation['bbox'][1]
        annotation['bbox'][1] = min(max(0, annotation['bbox'][1]), cropped_image_height)
        
        # Adjust bbox width
        annotation['bbox'][2] = min(max(0, annotation['bbox'][2] - (annotation['bbox'][0] - bbox_0_old)), cropped_image_width - annotation['bbox'][0])

        # Adjust bbox height
        annotation['bbox'][3] = min(max(0, annotation['bbox'][3] - (annotation['bbox'][1] - bbox_1_old)), cropped_image_height - annotation['bbox'][1])

        if annotation['bbox'][2] == 0 or annotation['bbox'][3] == 0:
            continue

        for i, keypoint in enumerate(annotation['keypoints']):
            if point_in_image(keypoint[0], keypoint[1], x_start, y_start, cropped_image_width, cropped_image_height):
                keypoint[0] -= x_start
                keypoint[1] -= y_start
            else:
                keypoint[0] = 0
                keypoint[1] = 0
                keypoint[2] = 0
            annotation['keypoints'][i] = keypoint

        LSD_annotations_cropped.append(annotation)
        
    return LSD_annotations_cropped

def convert_apollo_annotations_to_labelstudio_LSD(data, labelstudio_image_prefix, image_root, image_target_folder):
    LSD_labelstudio_data = []
    for image_data in data['images']:
        # Retrieve annotations for this image
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_data['id']]

        # Convert apollo annotations to LSD annotations
        LSD_annotations = convert_apollo_annotations_to_LSD(annotations)

        # Crop image to BeIntelli image size
        cropped_image_width = 1920
        cropped_image_height = 1208
        x_start = 732 # Center the crop
        y_start = 1200 # Move it a bit down until it barely captures the ego hood
        image = Image.open(image_root + image_data['file_name'])
        cropped_image = image.crop((x_start, y_start, x_start+cropped_image_width, y_start+cropped_image_height))

        # Save image
        cropped_image.save(image_target_folder + image_data['file_name'])

        # Adjust LSD annotations to cropped image
        LSD_annotations_cropped = adjust_LSD_annotations_to_cropped_image(LSD_annotations, x_start, y_start, cropped_image_width, cropped_image_height)

        # Convert LSD annotations to Label Studio predictions
        LSD_labelstudio_predictions = convert_LSD_annotations_to_labelstudio_predictions(LSD_annotations_cropped, cropped_image_width, cropped_image_height)

        # Append LSD annotations to labelstudio data
        LSD_labelstudio_data.append({
            "total_annotations": 1,
            "cancelled_annotations": 0,
            "id": str(image_data['id']),
            "data": {
                "image": labelstudio_image_prefix + image_data['file_name'],
            },
            "annotations": LSD_labelstudio_predictions
        })
    return LSD_labelstudio_data

def main():
    image_root =  './train/images/'
    labelstudio_image_prefix = '/data/local-files/?d=label-studio/data/apollo3d/images/'
    image_target_folder = 'images/'
    if not os.path.exists(image_target_folder):
        os.makedirs(image_target_folder)
    
    # Training data:
    apollo_annotation_file_train = './train/annotations/apollo_keypoints_66_train.json'
    labelstudio_LSD_annotation_file_train = 'labelstudio_LSD_annotations_train.json'

    data = load_apollo_3d(apollo_annotation_file_train)

    LSD_labelstudio_data = convert_apollo_annotations_to_labelstudio_LSD(data, labelstudio_image_prefix, image_root, image_target_folder)

    json.dump(LSD_labelstudio_data, open(labelstudio_LSD_annotation_file_train, 'w'))

    # Validation data:
    apollo_annotation_file_val = './train/annotations/apollo_keypoints_66_val.json'
    labelstudio_LSD_annotation_file_val = 'labelstudio_LSD_annotations_val.json'

    data = load_apollo_3d(apollo_annotation_file_val)

    LSD_labelstudio_data = convert_apollo_annotations_to_labelstudio_LSD(data, labelstudio_image_prefix, image_root, image_target_folder)

    json.dump(LSD_labelstudio_data, open(labelstudio_LSD_annotation_file_val, 'w'))

if __name__ == "__main__":
    main()
