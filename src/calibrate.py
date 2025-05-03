import argparse
import math

import cv2

# Parse arguments
parser = argparse.ArgumentParser(description="click endpoints to get y,h pairs")
parser.add_argument("--image", "-i", required=True, help="path to calibration frame")
args = parser.parse_args()

# load image
frame = cv2.imread(args.image)
if frame is None:
    print("failed to load image:", args.image)
    exit(1)

points = []  # clicked pts
measurements = []  # (y_mean, pixel_h) list


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"clicked: {x}, {y}")
        if len(points) == 2:
            (x0, y0), (x1, y1) = points
            h = math.hypot(x1 - x0, y1 - y0)
            y_mean = (y0 + y1) / 2.0
            measurements.append((y_mean, h))
            print(f"meas {len(measurements)} -> y={y_mean:.1f}, h={h:.1f}")
            points.clear()


cv2.namedWindow("calib")
cv2.setMouseCallback("calib", click)

print("click endpoints of D for far measurement")
print("then click endpoints of D for near measurement")
print("press esc to abort")

# main loop
while True:
    cv2.imshow("calib", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or len(measurements) >= 2:
        break

cv2.destroyAllWindows()

if len(measurements) < 2:
    print("need 2 measurements, exiting")
    exit(1)

(y1, h1), (y2, h2) = measurements

# ask for real distance
D = float(input("enter real-world span D (m): "))

# compute pix2m values
pix2m_1 = D / h1
pix2m_2 = D / h2

a = (pix2m_2 - pix2m_1) / (y2 - y1)
b = pix2m_1 - a * y1

# output
print("\nresults:")
print(f"y1 = {y1:.2f} px, h1 = {h1:.2f} px")
print(f"y2 = {y2:.2f} px, h2 = {h2:.2f} px")
print(f"pix2m_a (slope)     = {a:.6f}")
print(f"pix2m_b (intercept) = {b:.6f}")
