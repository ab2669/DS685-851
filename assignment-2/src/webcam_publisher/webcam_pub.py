#!/usr/bin/env python3
"""
Webcam Publisher Node for Assignment 2
Publishes webcam feed to /camera/image_raw topic

Author: Andrew Boksz | ab2669@njit.edu
Course: DS685-851 AI For Robotics
Assignment: 2
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class WebcamPublisher(Node):
    """
    ROS2 Node that captures webcam video and publishes it to a topic.
    """

    def __init__(self):
        """Initialize the webcam publisher node."""
        super().__init__('webcam_publisher')
        
        # Create publisher for camera images
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Set publishing rate (10 Hz = 10 frames per second)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Initialize webcam (0 = default camera)
        self.cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open webcam!')
            raise RuntimeError('Cannot open webcam')
        
        # Set camera resolution (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize CV Bridge for ROS-OpenCV conversion
        self.br = CvBridge()
        
        self.get_logger().info('Webcam Publisher started - publishing to /camera/image_raw')

    def timer_callback(self):
        """
        Timer callback function to capture and publish frames.
        Called at regular intervals defined by timer_period.
        """
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        
        if ret:
            # Convert OpenCV image to ROS Image message
            ros_image = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
            
            # Publish the image
            self.publisher_.publish(ros_image)
            
            # Log occasionally (every 50 frames to avoid spam)
            if self.get_clock().now().nanoseconds % 5000000000 < 100000000:
                self.get_logger().info('Publishing webcam frame')
        else:
            self.get_logger().warn('Failed to capture frame from webcam')

    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        self.cap.release()
        super().destroy_node()


def main(args=None):
    """Main function to initialize and run the node."""
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create the node
    webcam_publisher = WebcamPublisher()
    
    try:
        # Spin the node (keeps it running)
        rclpy.spin(webcam_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        webcam_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
