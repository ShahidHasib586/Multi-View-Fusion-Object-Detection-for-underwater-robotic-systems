#!/usr/bin/env python3
import os
import cv2
import json
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Stereo calibration from two synchronized chessboard videos.")
    p.add_argument("--left_video", required=True, help="Path to left camera video")
    p.add_argument("--right_video", required=True, help="Path to right camera video")
    p.add_argument("--board_cols", type=int, required=True, help="Number of INNER corners across")
    p.add_argument("--board_rows", type=int, required=True, help="Number of INNER corners down")
    p.add_argument("--square_size", type=float, required=True, help="Square size in meters, e.g. 0.025")
    p.add_argument("--frame_step", type=int, default=5, help="Use every Nth frame")
    p.add_argument("--max_pairs", type=int, default=50, help="Maximum accepted stereo pairs")
    p.add_argument("--out_dir", default="calib_out", help="Output directory")
    p.add_argument("--show", action="store_true", help="Show detections while processing")
    return p.parse_args()


def make_object_points(cols, rows, square_size):
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size
    return objp


def find_corners(gray, pattern_size):
    # First try the stronger SB detector
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    if ok and corners is not None:
        corners = corners.astype(np.float32)
        return True, corners

    # Fallback to classic detector
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return True, corners


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pattern_size = (args.board_cols, args.board_rows)
    objp = make_object_points(args.board_cols, args.board_rows, args.square_size)

    cap_l = cv2.VideoCapture(args.left_video)
    cap_r = cv2.VideoCapture(args.right_video)

    if not cap_l.isOpened():
        raise RuntimeError(f"Could not open left video: {args.left_video}")
    if not cap_r.isOpened():
        raise RuntimeError(f"Could not open right video: {args.right_video}")

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    frame_idx = 0
    pair_count = 0
    image_size = None

    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()

        if not ok_l or not ok_r:
            break

        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])

        found_l, corners_l = find_corners(gray_l, pattern_size)
        found_r, corners_r = find_corners(gray_r, pattern_size)

        disp_l = frame_l.copy()
        disp_r = frame_r.copy()

        if found_l:
            cv2.drawChessboardCorners(disp_l, pattern_size, corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(disp_r, pattern_size, corners_r, found_r)

        if found_l and found_r:
            objpoints.append(objp.copy())
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            pair_count += 1
            print(f"[INFO] accepted stereo pair {pair_count} at frame {frame_idx}")

        if args.show:
            both = np.hstack([disp_l, disp_r])
            cv2.putText(
                both,
                f"pairs={pair_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("stereo corners", both)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        if pair_count >= args.max_pairs:
            break

        frame_idx += 1

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

    if pair_count < 10:
        raise RuntimeError(
            f"Only found {pair_count} valid stereo pairs. "
            f"Need at least about 10, preferably 20 to 50."
        )

    print(f"[INFO] total accepted stereo pairs: {pair_count}")
    print("[INFO] calibrating left camera...")
    ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None
    )

    print("[INFO] calibrating right camera...")
    ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None
    )

    print(f"[INFO] left reprojection error:  {ret_l:.6f}")
    print(f"[INFO] right reprojection error: {ret_r:.6f}")

    flags = cv2.CALIB_FIX_INTRINSIC
    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    print("[INFO] stereo calibration...")
    ret_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        K1, D1,
        K2, D2,
        image_size,
        criteria=stereo_criteria,
        flags=flags
    )

    print(f"[INFO] stereo reprojection error: {ret_s:.6f}")
    print(f"[INFO] baseline length (meters): {float(np.linalg.norm(T)):.6f}")

    print("[INFO] stereo rectification...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    out_npz = os.path.join(args.out_dir, "stereo_calib.npz")
    np.savez(
        out_npz,
        image_width=image_size[0],
        image_height=image_size[1],
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y,
        map2x=map2x, map2y=map2y,
        left_error=ret_l,
        right_error=ret_r,
        stereo_error=ret_s
    )

    summary = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "num_pairs": int(pair_count),
        "left_reprojection_error": float(ret_l),
        "right_reprojection_error": float(ret_r),
        "stereo_reprojection_error": float(ret_s),
        "baseline_m": float(np.linalg.norm(T)),
        "T": T.reshape(-1).tolist()
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] wrote calibration file: {out_npz}")
    print(f"[INFO] wrote summary file: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
