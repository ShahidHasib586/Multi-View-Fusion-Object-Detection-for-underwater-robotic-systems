#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('video_path', '')
        self.declare_parameter('topic_name', '/video/image_raw')
        self.declare_parameter('fps', 30.0)

        video_path = self.get_parameter('video_path').value
        topic_name = self.get_parameter('topic_name').value
        fps = float(self.get_parameter('fps').value)

        if not os.path.exists(video_path):
            raise RuntimeError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video")

        self.publisher = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()

        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

        self.get_logger().info(f"Publishing {video_path} to {topic_name}")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Video finished. Restarting...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = VideoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()