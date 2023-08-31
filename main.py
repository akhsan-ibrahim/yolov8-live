import cv2
import argparse

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="YOLOv8 live")
  parser.add_argument(
    "--webcam-resolution",
    default=[1280,720],
    nargs=2,
    type=int
  )
  args = parser.parse_args()
  return args

def main():
  args = parse_arguments()
  frame_width, frame_height = args.webcam_resolution

  cap = cv2.VideoCapture(1)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

  while True:
    ret, frame = cap.read()
    cv2.imshow("yolov8", frame)

    # print(frame.shape)
    # break
    if (cv2.waitKey(30) == 27): # wait 30 ms for press Esc (27 ASCII)
      break

if __name__ == "__main__":
  main()
