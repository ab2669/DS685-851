#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

# Detectron2 imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')
        
        # Parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('publish_visualization', True)
        
        camera_topic = self.get_parameter('camera_topic').value
        conf_threshold = self.get_parameter('confidence_threshold').value
        self.publish_viz = self.get_parameter('publish_visualization').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Set up Detectron2 with COCO-pretrained Mask R-CNN
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        cfg.MODEL.DEVICE = 'cpu'  # Use CPU (change to 'cuda' if GPU available)
        
        self.predictor = DefaultPredictor(cfg)
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        
        self.get_logger().info(f'Detectron2 initialized with {len(self.class_names)} COCO classes')
        self.get_logger().info(f'Subscribing to: {camera_topic}')
        
        # Subscribe to camera topic
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detections', 10)
        
        if self.publish_viz:
            self.viz_pub = self.create_publisher(
                Image, '/detections/visualization', 10)
        
        self.frame_count = 0
        
    def image_callback(self, msg):
        self.frame_count += 1
        
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        # Run Detectron2 inference
        outputs = self.predictor(cv_image)
        
        # Extract detections
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        
        # Create Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        for box, cls, score in zip(boxes, classes, scores):
            detection = Detection2D()
            detection.header = msg.header
            
            # Bounding box
            x1, y1, x2, y2 = box
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Object hypothesis (class and confidence)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self.class_names[cls]
            hypothesis.hypothesis.score = float(score)
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        # Publish detections
        self.detection_pub.publish(detection_array)
        
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count}: Detected {len(boxes)} objects')
        
        # Visualization
        if self.publish_viz and len(boxes) > 0:
            viz_image = cv_image.copy()
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.class_names[cls]
                
                # Draw bounding box
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f'{class_name}: {score:.2f}'
                cv2.putText(viz_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Publish visualization
            try:
                viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding='bgr8')
                viz_msg.header = msg.header
                self.viz_pub.publish(viz_msg)
            except Exception as e:
                self.get_logger().error(f'Viz publishing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
