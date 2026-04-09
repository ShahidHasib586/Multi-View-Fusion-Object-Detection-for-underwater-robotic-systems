#!/usr/bin/env python3
import cv2
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Extract the same frame index from two synchronized videos.")
    p.add_argument("--left_video", required=True)
    p.add_argument("--right_video", required=True)
    p.add_argument("--frame_idx", type=int, required=True)
    p.add_argument("--left_out", default="left_frame.png")
    p.add_argument("--right_out", default="right_frame.png")
    return p.parse_args()


def read_frame(path, frame_idx):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
    return frame


def main():
    args = parse_args()
    left = read_frame(args.left_video, args.frame_idx)
    right = read_frame(args.right_video, args.frame_idx)
    cv2.imwrite(args.left_out, left)
    cv2.imwrite(args.right_out, right)
    print(f"Saved {args.left_out}")
    print(f"Saved {args.right_out}")


if __name__ == "__main__":
    main()
