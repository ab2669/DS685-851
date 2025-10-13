#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class RTSPCameraPublisher(Node):
    def __init__(self):
        super().__init__('rtsp_camera_publisher')
        
        # Create publisher for /camera/image_raw topic
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Initialize CV Bridge for converting OpenCV images to ROS messages
        self.bridge = CvBridge()
        
        # RTSP stream URL (use host.docker.internal for Mac->Docker communication)
        self.rtsp_url = "rtsp://host.docker.internal:8554/mystream"
        
        # Initialize OpenCV VideoCapture with RTSP stream
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Check if RTSP stream opened successfully
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open RTSP stream: {self.rtsp_url}')
            return
        else:
            self.get_logger().info(f'RTSP stream opened successfully: {self.rtsp_url}')
        
        # Create timer to publish at 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
    
    def timer_callback(self):
        # Capture frame from RTSP stream
        ret, frame = self.cap.read()
        
        if ret:
            try:
                # Convert OpenCV image (BGR) to ROS Image message
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = 'rtsp_camera'
                
                # Publish the image
                self.publisher.publish(img_msg)
                
            except Exception as e:
                self.get_logger().error(f'Error converting/publishing image: {str(e)}')
        else:
            self.get_logger().warn('Failed to capture frame from RTSP stream')
    
    def destroy_node(self):
        # Release camera when node is destroyed
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


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
