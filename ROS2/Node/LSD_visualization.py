import cv2
import torchvision

import numpy as np

label_to_text_array = [
                       "background",
                       "brake",
                       "brake+hazard",
                       "brake+left",
                       "brake+right",
                       "hazard",
                       "left",
                       "off",
                       "right"]

keypoints_classes_ids2color = []
keypoints_classes_ids2color_hex = ["#4e9a06", "#d99100", "#729fcf", "#fce94f", "#9c73bf", "#ff1a38", "#8ae234"]
for hex_color in keypoints_classes_ids2color_hex:
    hex_color_value = hex_color.lstrip('#')
    keypoints_classes_ids2color.append(tuple(int(hex_color_value[i:i+2], 16) for i in (0, 2, 4)))

keypoint_skeleton = [[1,2],[1,3],[2,7],[3,4],[4,5],[5,6],[5,7],[6,7]]

def draw_keypoint_skeleton(image, keypoints, keypoint_visibility_threshold, line_color=(0,255,128), line_thickness=2, keypoint_scores=None, keypoint_score_threshold=20):
    for kps_idx, kps in enumerate(keypoints):
        for edge in keypoint_skeleton:
            kp1 = kps[edge[0]-1]
            kp2 = kps[edge[1]-1]
            if kp1[2] > keypoint_visibility_threshold and kp2[2] > keypoint_visibility_threshold:
                if keypoint_scores is None or (keypoint_scores[kps_idx][edge[0]-1] > keypoint_score_threshold and keypoint_scores[kps_idx][edge[1]-1] > keypoint_score_threshold):
                    image = cv2.line(image.copy(), tuple(kp1[:2]), tuple(kp2[:2]), line_color, line_thickness)
    return image

def draw_light_state(image, kp_idx, state, keypoint):
    half_height = 4
    half_width = 4
    thickness = -1
    blink_color = (255,140,0)
    break_color = (255, 0, 0)
    off_color = (255, 255, 255)
    left_rect_color = off_color
    right_rect_color = off_color

    state_threshold = 0.5
    
    # "left_front_light",
    if kp_idx == 0:
        if state[1] > state_threshold:
            left_rect_color = blink_color
    # "left_mirror",
    elif kp_idx == 1:
        if state[1] > state_threshold:
            left_rect_color = blink_color
    # "right_front_light",
    elif kp_idx == 2:
        if state[1] > state_threshold:
            right_rect_color = blink_color
    # "right_mirror",
    elif kp_idx == 3:
        if state[1] > state_threshold:
            right_rect_color = blink_color
    # "right_rear_light",
    elif kp_idx == 4:
        if state[0] > state_threshold:
            left_rect_color = break_color
        if state[1] > state_threshold:
            right_rect_color = blink_color
    # "center_rear_light",
    elif kp_idx == 5:
        if state[0] > state_threshold:
            left_rect_color = break_color
            right_rect_color = break_color
    # "left_rear_light"
    elif kp_idx == 6:
        if state[0] > state_threshold:
            right_rect_color = break_color
        if state[1] > state_threshold:
            left_rect_color = blink_color

    # left rect
    image = cv2.rectangle(image, (keypoint[0]-half_width, keypoint[1]-half_height), (keypoint[0], keypoint[1]+half_height), left_rect_color, thickness)
    # right rect
    image = cv2.rectangle(image, (keypoint[0], keypoint[1]-half_height), (keypoint[0]+half_width, keypoint[1]+half_height), right_rect_color, thickness)
    return image
  
def visualize_from_output(image, output):
    scores = output[0]['scores'].detach().cpu().numpy()

    high_scores_idxs = np.where(scores > 0.5)[0].tolist() # Indexes of boxes with scores > 0.5
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
    
    keypoint_dimension = 3 # x,y,v
    
    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:keypoint_dimension])) for kp in kps])
    

    keypoint_scores = output[0]['keypoints_scores'][:][:].detach().cpu().numpy()
    labels = output[0]['labels'].detach().cpu().numpy().astype(np.int32).tolist()
    keypoint_states = output[0]['keypoint_states'].detach().cpu().numpy().tolist()
    
    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))
        
    return visualize_LS(image, bboxes, keypoints, labels, keypoint_states, keypoint_scores=keypoint_scores)


def visualize_LS(image, bboxes, keypoints, labels, keypoint_states, keypoint_scores=None, keypoint_score_threshold=10):
    keypoint_visibility_threshold = 0.5

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 1

    for bbox, label in zip(bboxes, labels):
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
        image = cv2.putText(image, label_to_text_array[label], start_point, font, fontScale, color, thickness, cv2.LINE_AA)
    
    image = draw_keypoint_skeleton(image.copy(), keypoints, keypoint_visibility_threshold, keypoint_scores=keypoint_scores, keypoint_score_threshold=keypoint_score_threshold)

    for kps_idx, kps in enumerate(keypoints):
        for idx, kp in enumerate(kps):
            if kp[2] > keypoint_visibility_threshold:
                if keypoint_scores is None or keypoint_scores[kps_idx][idx] > keypoint_score_threshold:
                    image = cv2.circle(image.copy(), tuple(kp[:2]), 4, keypoints_classes_ids2color[idx], 10)
                    image = draw_light_state(image, idx, keypoint_states[kps_idx][idx], kp[:2])
    
    return image.copy()
