#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('video_path', '')
        self.declare_parameter('topic_name', '/image_raw')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera')

        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.fps = float(self.get_parameter('fps').get_parameter_value().double_value)
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        if not self.video_path:
            raise RuntimeError("video_path is empty. Pass -p video_path:=/path/to/video.mp4")

        self.pub = self.create_publisher(Image, self.topic_name, 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        period = 1.0 / max(self.fps, 1e-3)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"Publishing {self.video_path} -> {self.topic_name} at {self.fps} FPS, frame_id={self.frame_id}")

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().info("Video ended. Looping to start.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VideoPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.cap.release()
            except Exception:
                pass
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
