
def _safe_names_from_ultralytics(detector):
    # Ultralytics may expose names on detector.names or detector.model.names
    try:
        if hasattr(detector, "names") and detector.names:
            return detector.names
    except Exception:
        pass
    try:
        if hasattr(detector, "model") and hasattr(detector.model, "names") and detector.model.names:
            return detector.model.names
    except Exception:
        pass
    return {}

import os
import time
from typing import List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import rclpy
from rclpy.node import Node

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from .enhance_model import load_enhancer

def cv_to_msg(node: Node, bridge: CvBridge, cv_img: np.ndarray, frame_id: str) -> Image:
    msg = bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    return msg

def make_detection_array(node: Node, detections, frame_id: str) -> Detection2DArray:
    out = Detection2DArray()
    out.header.stamp = node.get_clock().now().to_msg()
    out.header.frame_id = frame_id

    # detections is list of dicts: {xyxy, conf, cls}
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        conf = float(d["conf"])
        cls = int(d["cls"])

        det = Detection2D()
        det.header = out.header
        # If this detection came from fusion(), it may include a source camera index.
        # Tag it so /fusion/detections shows which camera contributed.
        if isinstance(d, dict) and ('cam' in d):
            cam_i = int(d.get('cam', -1))
            det.id = f"cam{cam_i}"
            det.header.frame_id = f"cam{cam_i}"

        bbox = BoundingBox2D()
        bbox.center.position.x = float((x1 + x2) / 2.0)
        bbox.center.position.y = float((y1 + y2) / 2.0)
        bbox.size_x = float(x2 - x1)
        bbox.size_y = float(y2 - y1)
        det.bbox = bbox

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(cls)
        hyp.hypothesis.score = conf
        det.results.append(hyp)

        out.detections.append(det)

    return out

class MultiCamFusionDetector(Node):

    def cb(self, *msgs):
        # message_filters ApproxTimeSync callback (N msgs)
        if len(msgs) == 1:
            return self.cb_single(msgs[0])
        return self.cb_multi(*msgs)

    def process_msgs(self, msgs):
        # Main processing path for 1..N cameras
        per_cam_dets = []

        for i, msg in enumerate(msgs):
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Enhance if model loaded, else pass-through
            if hasattr(self, "enhancer") and self.enhancer is not None:
                enh = self.enhance_bgr(bgr)
            else:
                enh = bgr

            dets, ann = self.run_yolo(enh)
            per_cam_dets.append(dets)

            # publish per-cam
            det_msg = make_detection_array(self, dets, frame_id=f"cam{i}")
            self.pub_det[i].publish(det_msg)
            self.pub_ann[i].publish(cv_to_msg(self, self.bridge, ann, frame_id=f"cam{i}"))

        fused = self.fuse(per_cam_dets) if hasattr(self, "fuse") else []
        fused_msg = make_detection_array(self, fused, frame_id="fusion")
        self.pub_fused.publish(fused_msg)


    def __init__(self):
        super().__init__("multicam_fusion_detector")
        self.bridge = CvBridge()

        self.declare_parameter("cam_topics", ["/cam0/image_raw", "/cam1/image_raw", "/cam2/image_raw"])
        self.declare_parameter("enhance_ckpt", "models/uie_best.pt")
        self.declare_parameter("enhance_img_size", 256)
        self.declare_parameter("enhance_ema", 0.90)

        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("yolo_imgsz", 640)
        self.declare_parameter("yolo_conf", 0.25)
        self.declare_parameter("yolo_iou", 0.45)
        self.declare_parameter("yolo_device", "cuda")

        self.declare_parameter("out_prefix", "/fusion")
        self.declare_parameter("sync_slop_sec", 0.07)

        self.cam_topics: List[str] = list(self.get_parameter("cam_topics").value)
        self.enhance_ckpt = self.get_parameter("enhance_ckpt").value
        self.enh_size = int(self.get_parameter("enhance_img_size").value)
        self.ema_alpha = float(self.get_parameter("enhance_ema").value)

        self.yolo_model_path = self.get_parameter("yolo_model").value
        self.yolo_imgsz = int(self.get_parameter("yolo_imgsz").value)
        self.yolo_conf = float(self.get_parameter("yolo_conf").value)
        self.yolo_iou = float(self.get_parameter("yolo_iou").value)
        self.yolo_device = self.get_parameter("yolo_device").value
        self.fp16 = (str(self.yolo_device).startswith("cuda") or str(self.yolo_device) == "cuda")
        self._frame_i = 0
        self.cuda_empty_cache_every = 200  # set 0 to disable

        self.out_prefix = self.get_parameter("out_prefix").value
        self.sync_slop = float(self.get_parameter("sync_slop_sec").value)

        # resolve ckpt path relative to package cwd
        if not os.path.isabs(self.enhance_ckpt):
            # package is run from install space; but relative paths work if we pass absolute in launch
            pass

        # enhancement model
        self.device = "cuda" if (torch.cuda.is_available() and "cuda" in self.yolo_device) else "cpu"
        self.get_logger().info(f"Using device={self.device}")
        # enhancement model (optional)
        if str(self.enhance_ckpt).strip():
            self.enhancer = load_enhancer(self.enhance_ckpt, self.device)
        else:
            self.enhancer = None
        self.enh_ema_gain = None
        self.enh_ema_gamma = None

        # detector
        self.detector = YOLO(self.yolo_model_path)

        self.class_names = _safe_names_from_ultralytics(self.detector)
        # Convert names to a dict[int,str] if needed
        if isinstance(self.class_names, (list, tuple)):
            self.class_names = {i: n for i, n in enumerate(self.class_names)}
        # publishers
        self.pub_ann = []
        self.pub_det = []
        for i in range(len(self.cam_topics)):
            self.pub_ann.append(self.create_publisher(Image, f"{self.out_prefix}/cam{i}/annotated", 10))
            self.pub_det.append(self.create_publisher(Detection2DArray, f"{self.out_prefix}/cam{i}/detections", 10))

        self.pub_fused = self.create_publisher(Detection2DArray, f"{self.out_prefix}/detections", 10)

        # multi-cam sync
        subs = [Subscriber(self, Image, t) for t in self.cam_topics]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=self.sync_slop)        # If only one camera topic, avoid message_filters and use sensor QoS
        if len(self.cam_topics) == 1:
            self.sub_single = self.create_subscription(
                Image, self.cam_topics[0], self.cb_single, qos_profile_sensor_data
            )
        else:
            self.sync.registerCallback(self.cb)
        try:
            self.get_logger().info("MultiCamFusionDetector started.")
            if hasattr(self, "cam_topics"):
                self.get_logger().info(f"cam_topics={self.cam_topics}")
            if hasattr(self, "enhance_ckpt"):
                self.get_logger().info(f"enhance_ckpt={self.enhance_ckpt}")
            if hasattr(self, "yolo_model_path"):
                self.get_logger().info(f"yolo={self.yolo_model_path}")
        except Exception as e:
            # never crash on logging
            pass

    def enhance_bgr(self, bgr: np.ndarray) -> np.ndarray:
        # resize to enhancer size for speed (then back)
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (self.enh_size, self.enh_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(self.device)

        y, gain, gamma = self.enhancer(x)

        # EMA smoothing (good for video stability)
        a = self.ema_alpha
        if a > 0.0:
            if self.enh_ema_gain is None:
                self.enh_ema_gain = gain.clone()
                self.enh_ema_gamma = gamma.clone()
            else:
                self.enh_ema_gain = a * self.enh_ema_gain + (1 - a) * gain
                self.enh_ema_gamma = a * self.enh_ema_gamma + (1 - a) * gamma

            y = torch.clamp(self.enh_ema_gain * x, 0.0, 1.0)
            y = torch.pow(y + 1e-6, self.enh_ema_gamma)
            y = torch.clamp(y, 0.0, 1.0)

        out = (y[0].permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out


    def run_yolo(self, bgr):
        # Ultralytics returns Results; we also draw labels using class names.
        results = self.detector.predict(
            source=bgr,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            device=0 if str(self.device).startswith("cuda") else "cpu",
            verbose=False
        )[0]

        dets = []
        if results.boxes is None:
            return dets, bgr

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes.xyxy is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []
        clss  = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else []

        import cv2
        ann = bgr.copy()
        for (x1, y1, x2, y2), c, k in zip(boxes, confs, clss):
            cls_i = int(k)
            name = self.class_names.get(cls_i, str(cls_i)) if hasattr(self, "class_names") else str(cls_i)

            dets.append({
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(c),
                "cls": cls_i,
                "name": name,
            })

            cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                ann,
                f"{name} {float(c):.2f}",
                (int(x1), max(0, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return dets, ann
    def fuse(self, per_cam_dets: List[List[dict]]) -> List[dict]:
        # Simple late fusion: concatenate, tag class id with camera index by adding offset in label space
        # (keeps it simple + doesn't assume overlap calibration)
        fused = []
        for cam_i, dets in enumerate(per_cam_dets):
            for d in dets:
                fused.append({"xyxy": d["xyxy"], "conf": d["conf"], "cls": int(d["cls"]), "cam": cam_i})
        return fused

    def cb_single(self, msg):
        # Single image processing path (NO recursion)
        return self.process_msgs([msg])

    def cb_multi(self, *msgs):
        return self.process_msgs(list(msgs))
def main():
    rclpy.init()
    node = MultiCamFusionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
