# pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117

import rclpy
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

import torch

from LSD_model import get_LSD_model
from LSD_utils import LSD_keypoint_nms
from LSD_visualization import visualize_from_output

from tier4_perception_msgs.msg import DetectedObjectsWithFeature, DetectedObjectWithFeature
from autoware_auto_perception_msgs.msg import ObjectClassification

import torchvision
from torchvision.transforms import functional

## For reference:

# LSD_LABEL_MAPPING = {
#     0: 0, # car
#     1: 0, # traffic sign
#     2: 6, # bicycle
#     3: 7, # pedestrian
#     4: 0, # traffic light
#     5: 0, # curb
#     6: 0, # other
#     7: 0, # unknown
# }

# AUTOWARE_LABEL_MAPPING = {
#     0: 1, # car
#     1: 0, # traffic sign
#     2: 6, # bicycle
#     3: 7, # pedestrian
#     4: 0, # traffic light
#     5: 0, # curb
#     6: 0, # other
#     7: 0, # unknown
# }

def apply_LSD_keypoint_nms(output, device, prob_threshold=0.7):
    for i in range(len(output)):
        output[i]['keypoints_scores'] = LSD_keypoint_nms(output[i]['keypoints_scores'], device, prob_threshold)
    return output

BRAKE_LABELS = [1, 2, 3, 4]
LEFT_LABELS = [2, 3, 5, 6]
RIGHT_LABELS = [2, 4, 5, 8]

class LSDNode(Node):
    def __init__(self):
        super().__init__('light_skeleton_detection_node')

        self.declare_parameter('input_topic', '/camera_front_wide/image_rect')
        self.declare_parameter('output_topic', '/perception/object_recognition/detection/rois0')
        self.declare_parameter('visualization_topic', '/camera_front_wide/lsd_image')

        _input_topic = self.get_parameter('input_topic')
        _output_topic = self.get_parameter('output_topic')
        _visualization_topic = self.get_parameter('visualization_topic')
	
        self.bridge = CvBridge()

        # Load model:
        weight_path = "./weights/LSD_weights.pth"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = get_LSD_model(weights_path=weight_path, device=self.device)
        self.model.to(self.device)
        self.model.eval()

        # Create a publisher for visualizing the output
        self.annotated_image_publisher = self.create_publisher(
            Image,
            _visualization_topic.value,
            10  # QoS profile depth
        )
	
	# Create publisher for the detected objects
        self.detection_publisher = self.create_publisher(
            DetectedObjectsWithFeature,
            _output_topic.value,
            1
        )

        # Create a subscriber to the input image topic
        self.image_subscription = self.create_subscription(
            Image,
            _input_topic.value,
            self.image_callback,
            10
        )
        self.image_subscription

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        image = functional.to_tensor(cv_image)
        images = [image.to(self.device)]
        
        with torch.no_grad():
            output = self.model(images)

        output_filtered = apply_LSD_keypoint_nms(output.copy(), self.device)

        self.publish_detection_from_output(output_filtered, msg)
        
        self.publish_visualization_image_from_output(output_filtered, cv_image, msg)

    def publish_detection_from_output(self, output, image_msg):
        msg = DetectedObjectsWithFeature()

        # copy the header
        header = image_msg.header
        msg.header = header
        
        # do BBox NMS:
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.5)[0].tolist() # Indexes of boxes with scores > 0.5
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
        
        
        obj_list = []
        bbox_labels_nms = output[0]["labels"][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
        for i, bbox in enumerate(output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()):
            # map the label, ignore the subclass
            label = 1 # We always assign car in our case
            probability = 1.0
            width = int(bbox[2] - bbox[0])
            height = int(bbox[3] - bbox[1])

            ## create new tier4 obj
            obj = DetectedObjectWithFeature()
             
            classification = ObjectClassification()
            classification.label = label
            classification.probability = probability

            obj.object.classification = [classification]
            
            # Add custom classifications
            # TODO: This is currently done only using the bbox labels.
            # A more advanced way with the information from the keypoint states is still in development
            if self.object_is_braking(bbox_labels_nms[i]):
                classification = ObjectClassification()
                classification.label = 100
                classification.probability = 1.0
                obj.object.classification.append(classification)

            if self.left_indicator_is_on(bbox_labels_nms[i]):
                classification = ObjectClassification()
                classification.label = 101
                classification.probability = 1.0
                obj.object.classification.append(classification)
            
            if self.right_indicator_is_on(bbox_labels_nms[i]):
                classification = ObjectClassification()
                classification.label = 102
                classification.probability = 1.0
                obj.object.classification.append(classification)
            
            obj.object.existence_probability = probability
            # always use shape of type bounding box (not cylinder or polygon)
            obj.object.shape.type = 0
            obj.object.shape.dimensions.x = float(height)
            obj.object.shape.dimensions.y = float(width)
            
            obj.feature.roi.x_offset = int(bbox[0])
            obj.feature.roi.y_offset = int(bbox[1])
            obj.feature.roi.height = height
            obj.feature.roi.width = width
            # TODO: check if correct
            obj.feature.roi.do_rectify = True

            obj_list.append(obj)
        
        msg.feature_objects = obj_list

        self.detection_publisher.publish(msg)

    def object_is_braking(self, label):
        if label in BRAKE_LABELS:
            return True
        return False
    
    def left_indicator_is_on(self, label):
        if label in LEFT_LABELS:
            return True
        return False
    
    def right_indicator_is_on(self, label):
        if label in RIGHT_LABELS:
            return True
        return False
    
    def publish_visualization_image_from_output(self, output, image, image_msg):
        image_out = visualize_from_output(image, output)
        annotated_image_msg = self.bridge.cv2_to_imgmsg(image_out, encoding='rgb8')
        annotated_image_msg.header = image_msg.header
        self.annotated_image_publisher.publish(annotated_image_msg)
    
def main(args=None):
    rclpy.init(args=args)

    lsd_node = LSDNode()

    rclpy.spin(lsd_node)

    lsd_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
