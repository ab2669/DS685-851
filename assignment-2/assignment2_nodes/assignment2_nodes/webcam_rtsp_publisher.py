#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RTSPCameraPublisher(Node):
    def __init__(self):
        super().__init__('rtsp_camera_publisher')
        
        # Declare parameters with defaults
        self.declare_parameter('rtsp_url', 'rtsp://host.docker.internal:8554/mystream')
        self.declare_parameter('topic_name', '/camera/image_raw')
        self.declare_parameter('frame_rate', 30)
        
        # Get parameter values
        rtsp_url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        
        # Create publisher for camera topic
        self.publisher = self.create_publisher(Image, topic_name, 10)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize OpenCV VideoCapture with RTSP/HTTP stream
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Check if stream opened successfully
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open stream: {self.rtsp_url}')
            self.get_logger().error('Make sure VLC or camera is streaming on the correct port')
            return
        else:
            self.get_logger().info(f'Stream opened successfully: {self.rtsp_url}')
            self.get_logger().info(f'Publishing to topic: {topic_name} at {frame_rate} Hz')
        
        # Create timer based on frame_rate parameter
        self.timer = self.create_timer(1.0 / frame_rate, self.timer_callback)
        self.frame_count = 0
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            # Convert OpenCV image to ROS Image message
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera'
                self.publisher.publish(msg)
                
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    self.get_logger().info(f'Published frame {self.frame_count}')
            except Exception as e:
                self.get_logger().error(f'Error converting/publishing frame: {str(e)}')
        else:
            self.get_logger().warn('Failed to read frame from stream')
    
    def __del__(self):
        # Release video capture when node is destroyed
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = RTSPCameraPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
