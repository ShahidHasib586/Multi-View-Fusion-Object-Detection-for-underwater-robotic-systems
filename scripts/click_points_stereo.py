#!/usr/bin/env python3
import cv2
import json
import argparse
import numpy as np


left_points = []
right_points = []


def parse_args():
    p = argparse.ArgumentParser(description="Click corresponding points in left/right images.")
    p.add_argument("--left_image", required=True)
    p.add_argument("--right_image", required=True)
    return p.parse_args()


def draw_points(img, points, color):
    out = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), 5, color, -1)
        cv2.putText(out, str(i), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def on_mouse_left(event, x, y, flags, param):
    global left_points
    if event == cv2.EVENT_LBUTTONDOWN:
        left_points.append([float(x), float(y)])


def on_mouse_right(event, x, y, flags, param):
    global right_points
    if event == cv2.EVENT_LBUTTONDOWN:
        right_points.append([float(x), float(y)])


def main():
    args = parse_args()

    img_l = cv2.imread(args.left_image)
    img_r = cv2.imread(args.right_image)

    if img_l is None:
        raise RuntimeError(f"Could not read {args.left_image}")
    if img_r is None:
        raise RuntimeError(f"Could not read {args.right_image}")

    cv2.namedWindow("left")
    cv2.namedWindow("right")
    cv2.setMouseCallback("left", on_mouse_left)
    cv2.setMouseCallback("right", on_mouse_right)

    while True:
        show_l = draw_points(img_l, left_points, (0, 255, 0))
        show_r = draw_points(img_r, right_points, (0, 255, 255))

        cv2.putText(show_l, "left: click points, s=save, u=undo, q=quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(show_r, "right: click same points in same order", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

        cv2.imshow("left", show_l)
        cv2.imshow("right", show_r)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u"):
            if len(left_points) > len(right_points):
                left_points.pop()
            elif len(right_points) > len(left_points):
                right_points.pop()
            else:
                if left_points:
                    left_points.pop()
                if right_points:
                    right_points.pop()

        elif key == ord("s"):
            if len(left_points) != len(right_points):
                print("Point counts differ. Click same number of points in both images.")
                continue
            with open("left_points.json", "w") as f:
                json.dump(left_points, f, indent=2)
            with open("right_points.json", "w") as f:
                json.dump(right_points, f, indent=2)
            print("Saved left_points.json and right_points.json")

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
