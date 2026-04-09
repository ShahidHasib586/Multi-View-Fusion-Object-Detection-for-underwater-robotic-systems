#!/usr/bin/env python3
import os
import cv2
import math
import json
import argparse
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Stereo YOLO triangulation from synchronized videos")
    p.add_argument("--left_video", required=True)
    p.add_argument("--right_video", required=True)
    p.add_argument("--calib", required=True, help="Path to stereo_calib.npz")
    p.add_argument("--yolo_model", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="cuda")
    p.add_argument("--match_dist_thresh", type=float, default=80.0,
                   help="Max distance in rectified image pixels between bbox centers to match same object")
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--save_json", default="triangulated_detections.json")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_calib(calib_path):
    data = np.load(calib_path)
    calib = {
        "K1": data["K1"],
        "D1": data["D1"],
        "K2": data["K2"],
        "D2": data["D2"],
        "R1": data["R1"],
        "R2": data["R2"],
        "P1": data["P1"],
        "P2": data["P2"],
        "map1x": data["map1x"],
        "map1y": data["map1y"],
        "map2x": data["map2x"],
        "map2y": data["map2y"],
    }
    return calib


def rectify_pair(left_bgr, right_bgr, calib):
    left_rect = cv2.remap(left_bgr, calib["map1x"], calib["map1y"], cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, calib["map2x"], calib["map2y"], cv2.INTER_LINEAR)
    return left_rect, right_rect


def bbox_center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_dist(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def run_yolo(detector, frame_bgr, imgsz, conf, iou, device):
    results = detector.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=0 if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu",
        verbose=False
    )[0]

    dets = []
    names = detector.names if hasattr(detector, "names") else {}

    if results.boxes is None:
        return dets

    boxes = results.boxes.xyxy.detach().cpu().numpy() if results.boxes.xyxy is not None else []
    confs = results.boxes.conf.detach().cpu().numpy() if results.boxes.conf is not None else []
    clss = results.boxes.cls.detach().cpu().numpy() if results.boxes.cls is not None else []

    for (x1, y1, x2, y2), s, c in zip(boxes, confs, clss):
        cls_id = int(c)
        name = names[cls_id] if isinstance(names, dict) and cls_id in names else str(cls_id)
        dets.append({
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "conf": float(s),
            "cls": cls_id,
            "name": name,
            "center": list(bbox_center_xyxy([float(x1), float(y1), float(x2), float(y2)])),
        })
    return dets


def match_detections(left_dets, right_dets, dist_thresh=80.0):
    matches = []
    used_right = set()

    for i, dl in enumerate(left_dets):
        best_j = -1
        best_dist = float("inf")

        for j, dr in enumerate(right_dets):
            if j in used_right:
                continue
            if dl["cls"] != dr["cls"]:
                continue

            d = point_dist(dl["center"], dr["center"])
            if d < dist_thresh and d < best_dist:
                best_dist = d
                best_j = j

        if best_j >= 0:
            used_right.add(best_j)
            matches.append((i, best_j, best_dist))

    return matches


def triangulate_center(pt_left, pt_right, calib):
    pts1 = np.array([[pt_left]], dtype=np.float32)   # shape (1,1,2)
    pts2 = np.array([[pt_right]], dtype=np.float32)

    pts1_rect = cv2.undistortPoints(
        pts1, calib["K1"], calib["D1"], R=calib["R1"], P=calib["P1"]
    ).reshape(-1, 2)

    pts2_rect = cv2.undistortPoints(
        pts2, calib["K2"], calib["D2"], R=calib["R2"], P=calib["P2"]
    ).reshape(-1, 2)

    X_h = cv2.triangulatePoints(
        calib["P1"], calib["P2"],
        pts1_rect.T, pts2_rect.T
    )

    X = (X_h[:3, :] / X_h[3, :]).reshape(3)
    return X


def draw_det(frame, det, color=(0, 255, 0), extra_text=""):
    x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
    label = f'{det["name"]} {det["conf"]:.2f}'
    if extra_text:
        label += f" | {extra_text}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, label, (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

    cx, cy = [int(v) for v in det["center"]]
    cv2.circle(frame, (cx, cy), 4, color, -1)


def main():
    args = parse_args()

    if not os.path.exists(args.left_video):
        raise FileNotFoundError(args.left_video)
    if not os.path.exists(args.right_video):
        raise FileNotFoundError(args.right_video)
    if not os.path.exists(args.calib):
        raise FileNotFoundError(args.calib)

    device = "cuda" if ("cuda" in args.device and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] using device: {device}")

    calib = load_calib(args.calib)
    detector = YOLO(args.yolo_model)

    cap_l = cv2.VideoCapture(args.left_video)
    cap_r = cv2.VideoCapture(args.right_video)
    if not cap_l.isOpened():
        raise RuntimeError(f"Could not open left video: {args.left_video}")
    if not cap_r.isOpened():
        raise RuntimeError(f"Could not open right video: {args.right_video}")

    frame_idx = 0
    all_results = []

    while True:
        ok_l, left = cap_l.read()
        ok_r, right = cap_r.read()
        if not ok_l or not ok_r:
            print("[INFO] end of one or both videos")
            break

        left_rect, right_rect = rectify_pair(left, right, calib)

        left_dets = run_yolo(detector, left_rect, args.imgsz, args.conf, args.iou, device)
        right_dets = run_yolo(detector, right_rect, args.imgsz, args.conf, args.iou, device)

        matches = match_detections(left_dets, right_dets, args.match_dist_thresh)

        left_vis = left_rect.copy()
        right_vis = right_rect.copy()

        frame_results = {
            "frame_idx": frame_idx,
            "matches": []
        }

        for li, rj, dist_px in matches:
            dl = left_dets[li]
            dr = right_dets[rj]

            X = triangulate_center(dl["center"], dr["center"], calib)
            Xv, Yv, Zv = float(X[0]), float(X[1]), float(X[2])

            fused_score = 1.0 - (1.0 - float(dl["conf"])) * (1.0 - float(dr["conf"]))

            frame_results["matches"].append({
                "class_id": int(dl["cls"]),
                "class_name": dl["name"],
                "left_conf": float(dl["conf"]),
                "right_conf": float(dr["conf"]),
                "fused_conf": float(fused_score),
                "left_center": dl["center"],
                "right_center": dr["center"],
                "match_distance_px": float(dist_px),
                "XYZ_m": [Xv, Yv, Zv],
            })

            draw_det(left_vis, dl, color=(0, 255, 0), extra_text=f"Z={Zv:.2f}m")
            draw_det(right_vis, dr, color=(0, 255, 255), extra_text=f"Z={Zv:.2f}m")

        # draw unmatched detections too
        matched_left = {li for li, _, _ in matches}
        matched_right = {rj for _, rj, _ in matches}

        for i, d in enumerate(left_dets):
            if i not in matched_left:
                draw_det(left_vis, d, color=(255, 0, 0), extra_text="unmatched")

        for j, d in enumerate(right_dets):
            if j not in matched_right:
                draw_det(right_vis, d, color=(255, 0, 0), extra_text="unmatched")

        all_results.append(frame_results)

        if args.show:
            both = np.hstack([left_vis, right_vis])
            cv2.putText(
                both,
                f"frame={frame_idx} matches={len(matches)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            cv2.imshow("stereo yolo triangulation", both)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if frame_idx % 10 == 0:
            print(f"[INFO] frame {frame_idx}: left={len(left_dets)} right={len(right_dets)} matches={len(matches)}")

        frame_idx += 1
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] saved results to {args.save_json}")


if __name__ == "__main__":
    main()
