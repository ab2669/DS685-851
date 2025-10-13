#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import json
from datetime import datetime

class VectorDBNode(Node):
    def __init__(self):
        super().__init__('vector_db_node')
        
        # Parameters
        self.declare_parameter('db_host', 'localhost')
        self.declare_parameter('db_port', 5432)
        self.declare_parameter('db_name', 'robotics')
        self.declare_parameter('db_user', 'postgres')
        self.declare_parameter('db_password', 'robotics123')
        
        # Get database connection parameters
        db_host = self.get_parameter('db_host').value
        db_port = self.get_parameter('db_port').value
        db_name = self.get_parameter('db_name').value
        db_user = self.get_parameter('db_user').value
        db_password = self.get_parameter('db_password').value
        
        # Connect to PostgreSQL
        try:
            self.conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                dbname=db_name,
                user=db_user,
                password=db_password
            )
            self.cursor = self.conn.cursor()
            self.get_logger().info('Connected to PostgreSQL database')
            
            # Initialize database schema
            self.init_database()
            
        except Exception as e:
            self.get_logger().error(f'Database connection failed: {e}')
            self.conn = None
            return
        
        # Robot pose tracking
        self.current_pose = None
        
        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        self.get_logger().info('Vector DB node initialized')
    
    def init_database(self):
        """Initialize PostgreSQL database with pgvector extension"""
        try:
            # Create pgvector extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create detections table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS object_detections (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    object_class TEXT NOT NULL,
                    confidence FLOAT,
                    bbox_center_x FLOAT,
                    bbox_center_y FLOAT,
                    bbox_width FLOAT,
                    bbox_height FLOAT,
                    robot_x FLOAT,
                    robot_y FLOAT,
                    robot_theta FLOAT,
                    embedding VECTOR(512),
                    metadata JSONB
                );
            """)
            
            # Create index for vector similarity search
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS embedding_idx 
                ON object_detections 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Create index on object class for fast lookups
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS class_idx 
                ON object_detections (object_class);
            """)
            
            self.conn.commit()
            self.get_logger().info('Database schema initialized')
            
        except Exception as e:
            self.get_logger().error(f'Database initialization failed: {e}')
            self.conn.rollback()
    
    def odom_callback(self, msg):
        """Track robot pose from odometry"""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }
    
    @staticmethod
    def quaternion_to_yaw(quat):
        """Convert quaternion to yaw angle"""
        import math
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def extract_embedding(self, detection):
        """
        Extract feature embedding from detection.
        For now, using a simple placeholder (bbox features).
        In production, use a CNN feature extractor (ResNet, CLIP, etc.)
        """
        # Placeholder: Create a 512-dim vector from bbox features
        # TODO: Replace with actual CNN embedding extraction
        embedding = np.random.randn(512).astype(np.float32)
        return embedding.tolist()
    
    def detection_callback(self, msg):
        """Process detections and store in database"""
        if self.conn is None:
            return
        
        if self.current_pose is None:
            self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        
        for detection in msg.detections:
            if len(detection.results) == 0:
                continue
            
            # Get detection info
            hypothesis = detection.results[0]
            object_class = hypothesis.hypothesis.class_id
            confidence = hypothesis.hypothesis.score
            
            # Get bounding box
            bbox_center_x = detection.bbox.center.position.x
            bbox_center_y = detection.bbox.center.position.y
            bbox_width = detection.bbox.size_x
            bbox_height = detection.bbox.size_y
            
            # Extract embedding (placeholder for now)
            embedding = self.extract_embedding(detection)
            
            # Create metadata
            metadata = {
                'frame_id': msg.header.frame_id,
                'timestamp_sec': msg.header.stamp.sec,
                'timestamp_nanosec': msg.header.stamp.nanosec
            }
            
            # Insert into database
            try:
                self.cursor.execute("""
                    INSERT INTO object_detections 
                    (object_class, confidence, bbox_center_x, bbox_center_y,
                     bbox_width, bbox_height, robot_x, robot_y, robot_theta,
                     embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    object_class, confidence,
                    bbox_center_x, bbox_center_y,
                    bbox_width, bbox_height,
                    self.current_pose['x'],
                    self.current_pose['y'],
                    self.current_pose['theta'],
                    embedding,
                    json.dumps(metadata)
                ))
                
                self.conn.commit()
                self.get_logger().info(
                    f"Stored {object_class} at ({self.current_pose['x']:.2f}, "
                    f"{self.current_pose['y']:.2f})")
                
            except Exception as e:
                self.get_logger().error(f'Database insert failed: {e}')
                self.conn.rollback()
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.cursor.close()
            self.conn.close()

def main(args=None):
    rclpy.init(args=args)
    node = VectorDBNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
