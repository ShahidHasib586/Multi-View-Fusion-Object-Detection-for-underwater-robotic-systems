#!/usr/bin/env python3
import json
import argparse
import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Triangulate matched 2D points using stereo calibration.")
    p.add_argument("--calib", required=True, help="Path to stereo_calib.npz")
    p.add_argument("--left_points_json", required=True, help="JSON list of [x,y] points in left image")
    p.add_argument("--right_points_json", required=True, help="JSON list of [x,y] points in right image")
    return p.parse_args()


def load_points(path):
    with open(path, "r") as f:
        pts = json.load(f)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Points in {path} must be Nx2")
    return pts


def main():
    args = parse_args()

    data = np.load(args.calib)

    K1 = data["K1"]
    D1 = data["D1"]
    K2 = data["K2"]
    D2 = data["D2"]
    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]

    pts1 = load_points(args.left_points_json)
    pts2 = load_points(args.right_points_json)

    if len(pts1) != len(pts2):
        raise RuntimeError("Left and right point files must have same number of points.")

    # OpenCV expects shape (N,1,2)
    pts1_u = pts1.reshape(-1, 1, 2)
    pts2_u = pts2.reshape(-1, 1, 2)

    # Undistort + rectify points
    pts1_rect = cv2.undistortPoints(pts1_u, K1, D1, R=R1, P=P1).reshape(-1, 2)
    pts2_rect = cv2.undistortPoints(pts2_u, K2, D2, R=R2, P=P2).reshape(-1, 2)

    # Triangulate
    points_4d = cv2.triangulatePoints(
        P1, P2,
        pts1_rect.T,
        pts2_rect.T
    )

    points_3d = (points_4d[:3, :] / points_4d[3, :]).T

    print("Triangulated 3D points (meters, stereo rectified frame):")
    for i, p in enumerate(points_3d):
        print(f"point_{i}: X={p[0]:.6f}, Y={p[1]:.6f}, Z={p[2]:.6f}")

    with open("triangulated_points.json", "w") as f:
        json.dump(points_3d.tolist(), f, indent=2)
    print("\nSaved to triangulated_points.json")


if __name__ == "__main__":
    main()
