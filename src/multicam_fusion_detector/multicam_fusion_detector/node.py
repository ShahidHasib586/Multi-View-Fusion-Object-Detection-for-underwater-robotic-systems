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
import math
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

    # detections is list of dicts: {xyxy, conf, cls, name, cam}
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        conf = float(d["conf"])
        cls = int(d["cls"])
        name = str(d.get("name", cls))

        det = Detection2D()
        det.header = out.header

        if isinstance(d, dict) and ("cam" in d):
            cam_val = d.get("cam", "fused")
            if isinstance(cam_val, int):
                det.id = f"cam{cam_val}"
                det.header.frame_id = f"cam{cam_val}"
            else:
                det.id = str(cam_val)
                det.header.frame_id = str(cam_val)

        bbox = BoundingBox2D()
        bbox.center.position.x = float((x1 + x2) / 2.0)
        bbox.center.position.y = float((y1 + y2) / 2.0)
        bbox.size_x = float(x2 - x1)
        bbox.size_y = float(y2 - y1)
        det.bbox = bbox

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = name
        hyp.hypothesis.score = conf
        det.results.append(hyp)

        out.detections.append(det)

    return out


class MultiCamFusionDetector(Node):

    def cb(self, *msgs):
        if len(msgs) == 1:
            return self.cb_single(msgs[0])
        return self.cb_multi(*msgs)

    def process_msgs(self, msgs):
        per_cam_dets = []

        for i, msg in enumerate(msgs):
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if hasattr(self, "enhancer") and self.enhancer is not None:
                enh = self.enhance_bgr(bgr)
            else:
                enh = bgr

            dets, ann = self.run_yolo(enh)
            per_cam_dets.append(dets)

            det_msg = make_detection_array(self, dets, frame_id=f"cam{i}")
            self.pub_det[i].publish(det_msg)
            self.pub_ann[i].publish(cv_to_msg(self, self.bridge, ann, frame_id=f"cam{i}"))

        fused = self.fuse(per_cam_dets) if hasattr(self, "fuse") else []
        fused_msg = make_detection_array(self, fused, frame_id="fusion")
        self.pub_fused.publish(fused_msg)

    def __init__(self):
        super().__init__("multicam_fusion_detector")
        self.bridge = CvBridge()

        self.declare_parameter("cam_topics", ["/cam0/image_raw", "/cam1/image_raw"])
        self.declare_parameter("enhance_ckpt", "")
        self.declare_parameter("enhance_img_size", 64)
        self.declare_parameter("enhance_ema", 0.90)

        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("yolo_imgsz", 416)
        self.declare_parameter("yolo_conf", 0.25)
        self.declare_parameter("yolo_iou", 0.45)
        self.declare_parameter("yolo_device", "cuda")

        self.declare_parameter("out_prefix", "/fusion")
        self.declare_parameter("sync_slop_sec", 0.15)
        self.declare_parameter("fusion_dist_thresh", 80.0)

        self.cam_topics: List[str] = list(self.get_parameter("cam_topics").value)
        self.enhance_ckpt = self.get_parameter("enhance_ckpt").value
        self.enh_size = int(self.get_parameter("enhance_img_size").value)
        self.ema_alpha = float(self.get_parameter("enhance_ema").value)

        self.yolo_model_path = self.get_parameter("yolo_model").value
        self.yolo_imgsz = int(self.get_parameter("yolo_imgsz").value)
        self.yolo_conf = float(self.get_parameter("yolo_conf").value)
        self.yolo_iou = float(self.get_parameter("yolo_iou").value)
        self.yolo_device = self.get_parameter("yolo_device").value

        self.out_prefix = self.get_parameter("out_prefix").value
        self.sync_slop = float(self.get_parameter("sync_slop_sec").value)
        self.fusion_dist_thresh = float(self.get_parameter("fusion_dist_thresh").value)

        self.device = "cuda" if (torch.cuda.is_available() and "cuda" in str(self.yolo_device)) else "cpu"
        self.get_logger().info(f"Using device={self.device}")

        if str(self.enhance_ckpt).strip():
            self.enhancer = load_enhancer(self.enhance_ckpt, self.device)
        else:
            self.enhancer = None

        self.enh_ema_gain = None
        self.enh_ema_gamma = None

        self.detector = YOLO(self.yolo_model_path)

        self.class_names = _safe_names_from_ultralytics(self.detector)
        if isinstance(self.class_names, (list, tuple)):
            self.class_names = {i: n for i, n in enumerate(self.class_names)}

        self.pub_ann = []
        self.pub_det = []
        for i in range(len(self.cam_topics)):
            self.pub_ann.append(self.create_publisher(Image, f"{self.out_prefix}/cam{i}/annotated", 10))
            self.pub_det.append(self.create_publisher(Detection2DArray, f"{self.out_prefix}/cam{i}/detections", 10))

        self.pub_fused = self.create_publisher(Detection2DArray, f"{self.out_prefix}/detections", 10)

        subs = [Subscriber(self, Image, t) for t in self.cam_topics]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=self.sync_slop)

        if len(self.cam_topics) == 1:
            self.sub_single = self.create_subscription(
                Image, self.cam_topics[0], self.cb_single, qos_profile_sensor_data
            )
        else:
            self.sync.registerCallback(self.cb)

        self.get_logger().info("MultiCamFusionDetector started.")
        self.get_logger().info(f"cam_topics={self.cam_topics}")
        self.get_logger().info(f"enhance_ckpt={self.enhance_ckpt}")
        self.get_logger().info(f"yolo={self.yolo_model_path}")

    def enhance_bgr(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (self.enh_size, self.enh_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            y, gain, gamma = self.enhancer(x)

            a = self.ema_alpha
            if a > 0.0:
                if self.enh_ema_gain is None:
                    self.enh_ema_gain = gain.detach()
                    self.enh_ema_gamma = gamma.detach()
                else:
                    self.enh_ema_gain = a * self.enh_ema_gain + (1 - a) * gain.detach()
                    self.enh_ema_gamma = a * self.enh_ema_gamma + (1 - a) * gamma.detach()

                y = torch.clamp(self.enh_ema_gain * x, 0.0, 1.0)
                y = torch.pow(y + 1e-6, self.enh_ema_gamma)
                y = torch.clamp(y, 0.0, 1.0)

        out = (y[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out

    def run_yolo(self, bgr):
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
        clss = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else []

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

    def _bbox_center_xyxy(self, xyxy):
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _bbox_distance_xyxy(self, b0, b1):
        c0 = self._bbox_center_xyxy(b0)
        c1 = self._bbox_center_xyxy(b1)
        return math.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)

    def fuse(self, per_cam_dets: List[List[dict]]) -> List[dict]:
        if len(per_cam_dets) == 0:
            return []

        if len(per_cam_dets) == 1:
            fused = []
            for d in per_cam_dets[0]:
                fused.append({
                    "xyxy": d["xyxy"],
                    "conf": d["conf"],
                    "cls": int(d["cls"]),
                    "name": d.get("name", str(int(d["cls"]))),
                    "cam": 0,
                })
            return fused

        cam0_dets = per_cam_dets[0]
        cam1_dets = per_cam_dets[1]

        fused = []
        used_cam1 = set()

        for d0 in cam0_dets:
            best_j = -1
            best_dist = float("inf")

            for j, d1 in enumerate(cam1_dets):
                if j in used_cam1:
                    continue

                if int(d0["cls"]) != int(d1["cls"]):
                    continue

                dist = self._bbox_distance_xyxy(d0["xyxy"], d1["xyxy"])
                if dist < self.fusion_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                d1 = cam1_dets[best_j]
                used_cam1.add(best_j)

                s0 = float(d0["conf"])
                s1 = float(d1["conf"])

                fused_score = 1.0 - (1.0 - s0) * (1.0 - s1)

                b0 = d0["xyxy"]
                b1 = d1["xyxy"]

                fused_bbox = [
                    (b0[0] + b1[0]) / 2.0,
                    (b0[1] + b1[1]) / 2.0,
                    (b0[2] + b1[2]) / 2.0,
                    (b0[3] + b1[3]) / 2.0,
                ]

                fused.append({
                    "xyxy": fused_bbox,
                    "conf": fused_score,
                    "cls": int(d0["cls"]),
                    "name": d0.get("name", str(int(d0["cls"]))),
                    "cam": "fused_cam0_cam1",
                })
            else:
                fused.append({
                    "xyxy": d0["xyxy"],
                    "conf": float(d0["conf"]),
                    "cls": int(d0["cls"]),
                    "name": d0.get("name", str(int(d0["cls"]))),
                    "cam": 0,
                })

        for j, d1 in enumerate(cam1_dets):
            if j not in used_cam1:
                fused.append({
                    "xyxy": d1["xyxy"],
                    "conf": float(d1["conf"]),
                    "cls": int(d1["cls"]),
                    "name": d1.get("name", str(int(d1["cls"]))),
                    "cam": 1,
                })

        for cam_i in range(2, len(per_cam_dets)):
            for d in per_cam_dets[cam_i]:
                fused.append({
                    "xyxy": d["xyxy"],
                    "conf": float(d["conf"]),
                    "cls": int(d["cls"]),
                    "name": d.get("name", str(int(d["cls"]))),
                    "cam": cam_i,
                })

        return fused

    def cb_single(self, msg):
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
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
